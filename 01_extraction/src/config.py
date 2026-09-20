from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """All run parameters in one place. Populated from CLI args in main.py.

    2026-07-29: switched from Qwen3.6-27B to Qwen3-32B. Qwen3.6 turned out
    to be a natively multimodal, hybrid-linear-attention architecture
    (shares code with Qwen3.5, not with the standard Qwen3 series) needing
    transformers >= 5.2/5.3 — and there is at least one documented vLLM
    compatibility issue on that exact architecture family from March 2026,
    unclear whether it's fixed in the pinned vllm==0.26.0 without testing on
    real hardware. Qwen3-32B is the original dense Qwen3 series: mature,
    long-supported by both transformers and vLLM, and it's the model
    --reasoning-parser qwen3 / enable_thinking / thinking_budget were built
    around in the first place, not a later adaptation. Same VRAM class at
    4-bit (~16-18GB), same single RTX 3090 target.
    """

    # 2026-08-13: switched to the official pre-quantized AWQ checkpoint
    # (see train.sh for why) — no on-the-fly bitsandbytes quantization
    # anymore, vLLM auto-detects AWQ from this repo's own config.json.
    model_name: str = "Qwen/Qwen3-32B-AWQ"
    base_url: str = "http://localhost:8000/v1"  # local vLLM OpenAI-compatible server
    api_key: str = "EMPTY"  # vLLM does not validate this

    dataset_source: str = "filippogreppi/Pile-NER-type-merged-CoT"
    split: str = "train"

    start_idx: int = 0
    end_idx: int | None = None  # None means "to the end of the dataset"

    # 2026-08-14: alternative to start_idx/end_idx for sharding — a path to
    # a text file (one id per line, from build_subsets.py) listing exactly
    # which remaining ids this job should process. Unlike a contiguous
    # index range, this correctly picks up rows that failed earlier
    # anywhere in the dataset, not just rows past wherever a previous run's
    # progress happened to stop. When set, takes priority over
    # start_idx/end_idx everywhere (main.py, output_path).
    id_subset_path: str | None = None

    # 2026-08-05: corrected from 0.7 to 0.6 — Qwen's own official
    # recommendation (HF model card, readthedocs quickstart) for THINKING
    # mode (which this pipeline uses) is Temperature=0.6, TopP=0.95,
    # TopK=20, MinP=0. 0.7 is their recommendation for NON-thinking mode
    # instead — the wrong one, used here by mistake until a tutor pointed
    # out to check the model's own recommended settings.
    temperature: float = 0.6
    # Coupled to --max-model-len in train.sh (4096 — see that file's
    # comment for the precise KV-cache-memory math behind this exact
    # value): max-model-len bounds prompt + completion TOGETHER, not
    # completion alone. The extraction prompt itself runs ~1500-1800
    # tokens, so 2048 leaves enough headroom below the 4096 ceiling. A
    # genuinely long reasoning case (ner_7, the Dragon Book parser
    # question) got truncated mid-thought at this same budget once
    # already — if that recurs often, the fix is raising
    # gpu_memory_utilization first (see train.sh), not just cranking this
    # number up in isolation.
    max_tokens: int = 2048
    # Caps the model's native <think> block so it cannot loop indefinitely
    # (documented Qwen3.x failure mode: some prompts never emit </think> and
    # burn the whole token budget on repeated reasoning, as observed on this
    # dataset with a blog post and even a plain code snippet). Counts toward
    # max_tokens, not on top of it. 1200 is a starting point calibrated on
    # the two real traces collected so far, not a validated final value —
    # tune it against pilot output.
    thinking_budget: int = 1200
    max_retries: int = 5
    request_timeout: float = 300.0

    # 2026-08-11: lowered from 24 to 8 after a real production run showed
    # this was badly oversubscribed. Server-side logs showed only 4-5
    # requests actually running concurrently at any time despite 24 being
    # requested, with 19-20 stuck waiting inside vLLM's own queue — a
    # direct consequence of --max-model-len 4096 leaving very little KV
    # cache (~4,800 tokens total) to actually run multiple sequences at
    # once. Requests stuck waiting still burn their 300s client timeout
    # before generation even starts, regardless of the text's content.
    # Concrete cost observed: 63% success rate (vs. 90% on every small
    # pilot), and a rate of 770 records per ~23 hours — a ~79-day
    # projected completion time for the full 45,889-row dataset. 8 is
    # closer to what the server actually sustains, not a generous
    # estimate — re-tune based on the next run's own "Running: N reqs"
    # log lines if it still doesn't match.
    concurrency: int = 8

    output_dir: str = "outputs"

    # Remote backup: pushes the accumulated output shard to a HF dataset repo
    # every hf_push_every results, so progress survives a lost node, not just
    # a killed process. Empty repo id disables it entirely.
    hf_backup_repo_id: str = ""
    hf_token: str = ""
    hf_push_every: int = 200

    # Stops attempting new extractions after this many consecutive failures
    # (likely means the vLLM server itself died) instead of retrying against
    # a dead server for the full backoff/timeout cycle on every remaining row.
    circuit_breaker_threshold: int = 15

    @property
    def output_path(self) -> str:
        """Shard filename encodes the index range (or subset filename), so
        parallel jobs on different nodes never collide (the cluster has no
        shared filesystem)."""
        if self.id_subset_path is not None:
            subset_stem = Path(self.id_subset_path).stem  # "subset_1" from ".../subset_1.txt"
            return os.path.join(self.output_dir, f"free_cot_{subset_stem}.jsonl")
        end_label = "end" if self.end_idx is None else str(self.end_idx)
        return os.path.join(self.output_dir, f"free_cot_{self.start_idx}_{end_label}.jsonl")
