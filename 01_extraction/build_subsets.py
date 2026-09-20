"""Computes which dataset ids have NOT yet been successfully completed, and
splits them into N subset files for parallel sharding.

Why this exists instead of just picking a contiguous index range to resume
from (SPLIT_START in the old sbatch_launch.sh): a contiguous range silently
never retries any row that failed earlier and was never completed — once a
later shard's range starts past that row's position, it's skipped forever.
This script instead computes the exact remaining id set (dataset ids minus
ids already present in the output file), so failed rows from anywhere in
the dataset get picked back up, not just rows past the last-seen position.

Usage:
    python3 build_subsets.py outputs/free_cot_0_45889.jsonl --n-subsets 4

Writes outputs/subset_1.txt ... outputs/subset_N.txt, one id per line.
Run each subset with:
    python3 main.py --id-subset-file outputs/subset_1.txt --model-name Qwen/Qwen3-32B-AWQ ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset import PileNERLoader


def load_done_ids(output_paths: list[Path]) -> set[str]:
    """Only ids with an actual result count as done — a line that failed to
    parse is skipped with a warning rather than silently treated as
    complete, same defensive stance as storage.py's own resume check."""
    done: set[str] = set()
    for path in output_paths:
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] {path}:{line_no} is not valid JSON, skipping (will be retried)")
                continue
            done.add(row["id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_paths", nargs="+", help="Existing .jsonl output file(s) to treat as already done")
    parser.add_argument("--n-subsets", type=int, default=4, help="Number of subset files to create (default: 4)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to write subset_*.txt files")
    parser.add_argument("--dataset-source", type=str, default="filippogreppi/Pile-NER-type-merged-CoT")
    args = parser.parse_args()

    output_paths = [Path(p) for p in args.output_paths]
    done_ids = load_done_ids(output_paths)
    print(f"{len(done_ids)} ids already completed, across {len(output_paths)} file(s)")

    loader = PileNERLoader(source=args.dataset_source)
    all_ids = loader.all_ids()
    remaining = [i for i in all_ids if i not in done_ids]
    print(f"{len(all_ids)} ids total in dataset, {len(remaining)} still remaining")

    if not remaining:
        print("Nothing left to do — every id is already completed.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ceiling-division chunking so every remaining id lands in exactly one
    # subset, and subsets differ in size by at most one id.
    n = args.n_subsets
    chunk_size = (len(remaining) + n - 1) // n
    for i in range(n):
        chunk = remaining[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            break
        subset_path = out_dir / f"subset_{i + 1}.txt"
        subset_path.write_text("\n".join(chunk) + "\n")
        print(f"  subset_{i + 1}.txt: {len(chunk)} ids")

    print(f"\nRun each with, e.g.:")
    print(f"  python3 main.py --id-subset-file {out_dir}/subset_1.txt --model-name Qwen/Qwen3-32B-AWQ "
          f"--hf-backup-repo-id \"$HF_BACKUP_REPO_ID\"")


if __name__ == "__main__":
    main()
