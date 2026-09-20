"""Step 1 entry point: free-CoT generation over a Pile-NER-type shard.

Designed for long, unattended runs on the cluster. Three layers of
resilience on top of the basic producer/consumer pipeline:

- Local JSONL is the source of truth during the run (immediate, cheap).
- A periodic backup push to a Hugging Face dataset repo means progress
  survives even if the node itself is lost, not just a killed process; on
  startup, if the local shard is missing, it is recovered from there first.
- A circuit breaker stops attempting new extractions after too many
  consecutive failures (the server is very likely dead), instead of
  retrying against nothing for the full backoff/timeout cycle on every
  remaining row.
- SIGTERM (SLURM sends this before SIGKILL when a job hits its time limit)
  triggers one last forced checkpoint push before the process dies.

Safe to re-submit after any interruption: already processed ids (local or
recovered from the Hub) are skipped automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

import colorlog
from dotenv import load_dotenv
from tqdm import tqdm

from src.circuit_breaker import FailureCircuitBreaker
from src.config import Config
from src.dataset import PileNERLoader, PileNERRecord
from src.extractor import FreeCoTExtractor
from src.hf_sync import HFCheckpointSync
from src.llm_client import AsyncOpenAICompatibleClient
from src.storage import JSONLResultWriter

logger = logging.getLogger("free_cot")


def setup_logging() -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--id-subset-file", type=str, default=None,
                         help="Path to a subset_N.txt from build_subsets.py (one id per line). "
                              "Takes priority over --start-idx/--end-idx when set.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--dataset-source", type=str, default="filippogreppi/Pile-NER-type-merged-CoT")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking-budget", type=int, default=1200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--hf-backup-repo-id", type=str, default="",
                         help="e.g. yourname/free-cot-checkpoints. Empty disables remote backup.")
    parser.add_argument("--hf-push-every", type=int, default=200)
    parser.add_argument("--circuit-breaker-threshold", type=int, default=15)
    return parser.parse_args()


async def produce(
    records: list[PileNERRecord],
    extractor: FreeCoTExtractor,
    semaphore: asyncio.Semaphore,
    queue: asyncio.Queue,
    breaker: FailureCircuitBreaker,
) -> None:
    """Runs extractions concurrently (bounded by semaphore), pushes each
    finished result (or error) onto the queue as soon as it completes."""

    async def run_one(record: PileNERRecord) -> None:
        async with semaphore:
            if breaker.is_open:
                # Fast-fail: skip the actual call entirely rather than
                # burning a full retry/backoff/timeout cycle against a
                # server that almost certainly is not coming back this run.
                await queue.put(("error", record.id, "circuit breaker open — too many consecutive failures"))
                return
            try:
                result = await extractor.extract(record.id, record.span, record.gold)
                breaker.record_success()
                await queue.put(("ok", record.id, result))
            except Exception as error:
                breaker.record_failure()
                await queue.put(("error", record.id, error))

    await asyncio.gather(*(run_one(r) for r in records))
    await queue.put(("done", None, None))  # sentinel: producer finished


async def consume(
    queue: asyncio.Queue,
    writer: JSONLResultWriter,
    sync: HFCheckpointSync,
    output_path: Path,
    total: int,
    background_pushes: list[asyncio.Task],
) -> tuple[int, int]:
    """Drains the queue, writes successful results to disk, triggers
    periodic remote backups, logs failures. Background push tasks are
    collected in `background_pushes` so the caller can await them cleanly
    instead of letting them potentially get abandoned when the event loop
    winds down."""
    n_success, n_failed = 0, 0
    with tqdm(total=total, desc="Extracting") as bar:
        while True:
            kind, record_id, payload = await queue.get()
            if kind == "done":
                break
            if kind == "ok":
                writer.append(payload)
                if sync.should_push():
                    # Offloaded to a thread: push_now() does a blocking HTTP
                    # upload, and running it inline here would freeze every
                    # in-flight extraction for the duration of the upload.
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
        model=config.model_name,
        base_url=config.base_url,
        api_key=config.api_key,
        max_retries=config.max_retries,
        request_timeout=config.request_timeout,
    )
    extractor = FreeCoTExtractor(
        client, temperature=config.temperature, max_tokens=config.max_tokens, thinking_budget=config.thinking_budget
    )
    loader = PileNERLoader(source=config.dataset_source, split=config.split)
    writer = JSONLResultWriter(config.output_path)
    breaker = FailureCircuitBreaker(threshold=config.circuit_breaker_threshold)

    already_done = writer.already_processed_ids()
    if already_done:
        logger.info("Resuming: %d records already present in %s, will be skipped.", len(already_done), output_path)

    all_records = (
        list(loader.iter_records_by_ids(set(Path(config.id_subset_path).read_text().split())))
        if config.id_subset_path is not None
        else list(loader.iter_records(config.start_idx, config.end_idx))
    )
    pending = [r for r in all_records if r.id not in already_done]
    shard_label = config.id_subset_path if config.id_subset_path is not None else f"[{config.start_idx}:{config.end_idx if config.end_idx else 'end'}]"
    logger.info("Shard %s: %d records total, %d pending, concurrency=%d.",
                shard_label, len(all_records), len(pending), config.concurrency)

    if not pending:
        logger.info("Nothing to do, shard already complete.")
        sync.push_now(output_path)
        return

    semaphore = asyncio.Semaphore(config.concurrency)
    queue: asyncio.Queue = asyncio.Queue()
    background_pushes: list[asyncio.Task] = []

    # SLURM sends SIGTERM before SIGKILL when a job hits its time limit.
    # This does not attempt to gracefully drain in-flight requests (too
    # little time is usually given for that) — it just makes sure whatever
    # has already landed on disk gets one last push to the remote backup.
    loop = asyncio.get_running_loop()

    def handle_sigterm() -> None:
        logger.warning("SIGTERM received (likely SLURM time limit) — pushing final checkpoint.")
        sync.push_now(output_path)

    loop.add_signal_handler(signal.SIGTERM, handle_sigterm)

    try:
        producer = asyncio.create_task(produce(pending, extractor, semaphore, queue, breaker))
        n_success, n_failed = await consume(queue, writer, sync, output_path, len(pending), background_pushes)
        await producer

        if breaker.is_open:
            logger.error("Circuit breaker tripped during this run — the vLLM server likely died. "
                          "Check vllm_server.log before re-submitting.")

        logger.info("Done. %d succeeded, %d failed, output: %s", n_success, n_failed, output_path)
    finally:
        # Runs even on an unhandled exception escaping the block above, so a
        # crash never silently drops the last stretch of already-generated
        # results that a periodic push hadn't caught yet.
        if background_pushes:
            await asyncio.gather(*background_pushes, return_exceptions=True)
        sync.push_now(output_path)  # unconditional final flush, regardless of the push_every counter


def main() -> None:
    setup_logging()
    load_dotenv()
    args = parse_args()

    config = Config(
        model_name=args.model_name,
        base_url=args.base_url,
        dataset_source=args.dataset_source,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        id_subset_path=args.id_subset_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_budget=args.thinking_budget,
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
