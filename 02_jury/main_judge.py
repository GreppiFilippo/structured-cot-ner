"""Step 2 entry point: one judge model scores every row of an extraction
output file (e.g. outputs/free_cot_merged.jsonl from Step 1).

Mirrors main.py's resilience design exactly — same three layers, just
applied to judging instead of extracting:

- Local JSONL is the source of truth during the run.
- A periodic backup push to a Hugging Face dataset repo means progress
  survives even if the node itself is lost; on startup, if the local
  shard is missing, it is recovered from there first.
- A circuit breaker stops attempting new judgments after too many
  consecutive failures instead of retrying against a dead server.
- SIGTERM triggers one last forced checkpoint push before the process dies.

Run three times (once per judge, --judge-model set differently each time)
against the SAME input file — see sbatch_launch_judge.sh, which submits
all three as parallel jobs. Safe to re-submit any single judge's job after
an interruption: already-scored ids in that judge's own output are
skipped automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from pathlib import Path

import colorlog
from dotenv import load_dotenv
from tqdm import tqdm

from src.circuit_breaker import FailureCircuitBreaker
from src.config import Config
from src.hf_sync import HFCheckpointSync
from src.judge_parser import JudgeScore, parse_judge_response
from src.judge_prompts import build_jury_prompt
from src.llm_client import AsyncOpenAICompatibleClient
from src.storage import JSONLResultWriter

logger = logging.getLogger("jury")


def setup_logging() -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-model", type=str, required=True,
                         help="e.g. google/gemma-3-4b-it, deepseek-ai/DeepSeek-R1-Distill-Llama-8B, "
                              "ibm-granite/granite-4.1-8b")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--input-path", type=str, required=True,
                         help="Extraction output .jsonl to score, e.g. outputs/free_cot_merged.jsonl")
    parser.add_argument("--id-subset-file", type=str, default=None,
                         help="Path to a subset_N.txt file (one id per line, from build_judge_subsets.py) — "
                              "only score rows whose id appears in this file. Omit to score the whole "
                              "--input-path, same as before this flag existed.")
    parser.add_argument("--limit", type=int, default=None, help="Only score the first N rows (for smoke-testing)")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--top-p", type=float, default=None,
                         help="Omit to let vLLM/model default apply. Check the judge's own model card — "
                              "recommended values differ per model (see src/config.py's comments).")
    parser.add_argument("--top-k", type=int, default=None, help="Same as --top-p: omit unless the judge's own card recommends a value.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--hf-backup-repo-id", type=str, default="",
                         help="e.g. yourname/jury-scores. Empty disables remote backup.")
    parser.add_argument("--hf-push-every", type=int, default=200)
    parser.add_argument("--circuit-breaker-threshold", type=int, default=15)
    return parser.parse_args()


def load_rows_to_score(input_path: str, limit: int | None, id_subset_file: str | None = None) -> list[dict]:
    subset_ids: set[str] | None = None
    if id_subset_file:
        with open(id_subset_file) as f:
            subset_ids = {line.strip() for line in f if line.strip()}

    rows = []
    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if subset_ids is not None and row["id"] not in subset_ids:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


async def produce(
    rows: list[dict],
    client: AsyncOpenAICompatibleClient,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    queue: asyncio.Queue,
    breaker: FailureCircuitBreaker,
    top_p: float | None = None,
    top_k: int | None = None,
) -> None:
    """Same structure as main.py's produce() — one judge call per row,
    bounded by semaphore, results pushed onto the queue as they land."""

    async def score_one(row: dict) -> None:
        async with semaphore:
            if breaker.is_open:
                await queue.put(("error", row["id"], "circuit breaker open — too many consecutive failures"))
                return
            try:
                prompt = build_jury_prompt(row)
                raw = await client.complete(
                    prompt, temperature=temperature, max_tokens=max_tokens, thinking_budget=0,
                    top_p=top_p, top_k=top_k,
                )
                score = parse_judge_response(row["id"], judge_model, raw)
                if score is None:
                    raise ValueError(f"no parseable JSON in judge output: {raw[:300]}")
                breaker.record_success()
                await queue.put(("ok", row["id"], score))
            except Exception as error:
                breaker.record_failure()
                await queue.put(("error", row["id"], error))

    await asyncio.gather(*(score_one(r) for r in rows))
    await queue.put(("done", None, None))


async def consume(
    queue: asyncio.Queue,
    writer: JSONLResultWriter,
    sync: HFCheckpointSync,
    output_path: Path,
    total: int,
    background_pushes: list[asyncio.Task],
) -> tuple[int, int]:
    n_success, n_failed = 0, 0
    with tqdm(total=total, desc="Judging") as bar:
        while True:
            kind, record_id, payload = await queue.get()
            if kind == "done":
                break
            if kind == "ok":
                writer.append(payload)
                if sync.should_push():
                    background_pushes.append(asyncio.create_task(asyncio.to_thread(sync.push_now, output_path)))
                n_success += 1
            else:
                logger.warning("Failed on %s: %s", record_id, payload)
                n_failed += 1
            bar.update(1)
    return n_success, n_failed


async def run(config: Config) -> None:
    output_path = Path(config.output_path)
    sync = HFCheckpointSync(
        repo_id=config.hf_backup_repo_id, token=config.hf_token or None, push_every=config.hf_push_every
    )
    if sync.enabled:
        logger.info("Remote backup enabled: %s (push every %d results)", config.hf_backup_repo_id, config.hf_push_every)
        sync.try_recover(output_path)
    else:
        logger.info("Remote backup disabled (no --hf-backup-repo-id). Local file only.")

    client = AsyncOpenAICompatibleClient(
        model=config.judge_model,
        base_url=config.base_url,
        api_key=config.api_key,
        max_retries=config.max_retries,
        request_timeout=config.request_timeout,
    )
    writer = JSONLResultWriter(config.output_path)
    breaker = FailureCircuitBreaker(threshold=config.circuit_breaker_threshold)

    already_done = writer.already_processed_ids()
    if already_done:
        logger.info("Resuming: %d rows already scored in %s, will be skipped.", len(already_done), output_path)

    all_rows = load_rows_to_score(config.input_path, config.limit, config.id_subset_file)
    pending = [r for r in all_rows if r["id"] not in already_done]
    logger.info("Judge %s: %d rows total, %d pending, concurrency=%d.",
                config.judge_model, len(all_rows), len(pending), config.concurrency)

    if not pending:
        logger.info("Nothing to do, this judge's pass is already complete.")
        sync.push_now(output_path)
        return

    semaphore = asyncio.Semaphore(config.concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    background_pushes: list[asyncio.Task] = []

    loop = asyncio.get_running_loop()

    def handle_sigterm() -> None:
        logger.warning("SIGTERM received (likely SLURM time limit) — pushing final checkpoint.")
        sync.push_now(output_path)

    loop.add_signal_handler(signal.SIGTERM, handle_sigterm)

    try:
        producer = asyncio.create_task(
            produce(pending, client, config.judge_model, config.temperature, config.max_tokens,
                    semaphore, queue, breaker, top_p=config.top_p, top_k=config.top_k)
        )
        n_success, n_failed = await consume(queue, writer, sync, output_path, len(pending), background_pushes)
        await producer

        if breaker.is_open:
            logger.error("Circuit breaker tripped during this run — the vLLM server likely died. "
                          "Check vllm_server.log before re-submitting.")

        logger.info("Done. %d succeeded, %d failed, output: %s", n_success, n_failed, output_path)
    finally:
        if background_pushes:
            await asyncio.gather(*background_pushes, return_exceptions=True)
        sync.push_now(output_path)


def main() -> None:
    setup_logging()
    load_dotenv()
    args = parse_args()

    config = Config(
        judge_model=args.judge_model,
        base_url=args.base_url,
        input_path=args.input_path,
        id_subset_file=args.id_subset_file,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        hf_backup_repo_id=args.hf_backup_repo_id,
        hf_token=os.environ.get("HF_TOKEN", ""),
        hf_push_every=args.hf_push_every,
        circuit_breaker_threshold=args.circuit_breaker_threshold,
    )
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
