from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """All run parameters for one judge's evaluation pass. Populated from
    CLI args in main_judge.py.

    2026-08-14: one run = one judge model scoring every row of a given
    extraction output file. Unlike the extractor (which shards by dataset
    index/id), sharding here is by JUDGE — three judges, each its own
    sbatch job on its own GPU, all reading the same input file and each
    writing its own output. No AWQ/GPTQ checkpoint verified available for
    all three judge models (unlike Qwen3-32B-AWQ, which Qwen publishes
    officially) — but at 4-8B parameters each, bf16 fits comfortably on a
    single 24GB GPU without quantization at all (Gemma-3-4B: ~8.6GB,
    DeepSeek-R1-Distill-Llama-8B / Granite-4.1-8B: ~16GB each), so this
    sidesteps the question entirely rather than hunting for or trusting an
    unofficial quantized checkpoint.
    """

    judge_model: str = ""  # set explicitly per job — no single sensible default across 3 different judges
    base_url: str = "http://localhost:8000/v1"  # local vLLM OpenAI-compatible server
    api_key: str = "EMPTY"  # vLLM does not validate this

    input_path: str = ""  # extraction output file to score, e.g. outputs/free_cot_merged.jsonl
    id_subset_file: str | None = None  # path to a subset_N.txt from build_judge_subsets.py — None means score the whole input_path
    limit: int | None = None  # process only the first N rows — for smoke-testing a judge before a full run

    # 2026-08-29: these are launch-time defaults only — checked each
    # judge's own official model card and their recommended values
    # genuinely differ, so the actual per-judge values are passed via CLI
    # flags at launch (see sbatch_launch_judge.sh / the individual sbatch
    # commands), not left at whatever sits here:
    #   Gemma-3-4B-it:            temperature=1.0, top_p=0.95, top_k=64
    #   DeepSeek-R1-Distill-8B:   temperature=0.6, top_p=0.95,
    #                             max_tokens=8192 (their card states a
    #                             32,768-token generation length as
    #                             normal for this family — our 1024
    #                             default was 32x too small, and the
    #                             direct cause of the "no parseable JSON,
    #                             ran out of tokens mid-reasoning"
    #                             failures seen in testing; 8192 is a
    #                             practical middle ground, not the full
    #                             32,768, since a judge call happening
    #                             thousands of times doesn't want to pay
    #                             for the rare worst case every time)
    #   Granite-4.1-8B:           temperature=1.0, top_p=0.95 (per the
    #                             wider Granite 4.x family's own stated
    #                             recommendation, applied "across all
    #                             tasks including reasoning")
    # These are the model creators' general-purpose recommendations, not
    # necessarily optimal for a low-variance SCORING task specifically —
    # worth revisiting if judge output quality still looks off after
    # applying them, rather than assuming they're automatically correct
    # here just because they're official.
    temperature: float = 0.3
    max_tokens: int = 1024  # override per-judge at launch — see note above, especially for DeepSeek
    top_p: float | None = None  # None = omit from request, let vLLM/model default apply
    top_k: int | None = None  # None = omit from request, let vLLM/model default apply
    max_retries: int = 5
    request_timeout: float = 300.0
    concurrency: int = 8  # same starting point as the extractor's own tuned value — re-tune from this run's own logs if it doesn't match

    output_dir: str = "outputs"

    hf_backup_repo_id: str = ""
    hf_token: str = ""
    hf_push_every: int = 200

    circuit_breaker_threshold: int = 15

    @property
    def output_path(self) -> str:
        """Filename encodes which judge produced it — three judges reading
        the SAME input file each need a distinct output, or they'd
        overwrite each other (no shared filesystem across nodes either).

        2026-08-31: when id_subset_file is set, the subset's own filename
        stem is folded into the output name too (e.g. jury_scores_gemma-
        3-4b-it_subset_2.jsonl) — running the SAME judge on 4 subsets in
        parallel (4 sbatch jobs) would otherwise all resolve to the exact
        same output_path and silently overwrite each other's results,
        each job only ever seeing a quarter of what it should. Every
        subset needs its own file so the eventual merge_judge_parts.py
        pass sees all of them and every row is scored by that judge
        exactly once, not zero or several times.
        """
        judge_label = self.judge_model.split("/")[-1].replace(".", "-")
        if self.id_subset_file:
            subset_label = os.path.splitext(os.path.basename(self.id_subset_file))[0]
            return os.path.join(self.output_dir, f"jury_scores_{judge_label}_{subset_label}.jsonl")
        return os.path.join(self.output_dir, f"jury_scores_{judge_label}.jsonl")
