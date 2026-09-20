"""Computes which rows a given judge has NOT yet scored, and splits the
remainder into N subset files for parallel processing — works for any of
the three judges (Gemma, DeepSeek, Granite), on any platform (SLURM/
Docker, Kaggle, Colab): the logic only ever deals with row ids and
generic "done" jsonl files, nothing judge- or platform-specific.

Same "exact remaining id set" logic as full_test_cot/build_subsets.py —
a contiguous range would silently never retry a row that failed midway
through an earlier session; this instead computes done ids (from any
existing jury_scores_<judge>_part*.jsonl output) minus the full id set,
so failures from anywhere get picked back up.

Usage:
    python3 build_judge_subsets.py data/free_cot_merged.jsonl \\
        outputs/jury_scores_gemma-3-4b-it_part*.jsonl --n-subsets 4

    python3 build_judge_subsets.py data/free_cot_merged.jsonl \\
        outputs/jury_scores_DeepSeek-R1-Distill-Llama-8B.jsonl --n-subsets 2

Writes subset_1.txt ... subset_N.txt, one id per line, in --output-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_done_ids(output_paths: list[Path]) -> set[str]:
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


def load_all_ids(input_path: Path) -> list[str]:
    ids = []
    for line_no, line in enumerate(input_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [warn] {input_path}:{line_no} is not valid JSON, skipping")
            continue
        ids.append(row["id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_path", help="Path to free_cot_merged.jsonl")
    parser.add_argument("done_paths", nargs="*",
                         help="Existing jury_scores_gemma*.jsonl output file(s) to treat as already done "
                              "(pass any you already have, e.g. from earlier Kaggle/Colab sessions)")
    parser.add_argument("--n-subsets", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    done_paths = [Path(p) for p in args.done_paths]

    done_ids = load_done_ids(done_paths)
    print(f"{len(done_ids)} ids already scored, across {len(done_paths)} file(s)")

    all_ids = load_all_ids(input_path)
    remaining = [i for i in all_ids if i not in done_ids]
    print(f"{len(all_ids)} ids total in dataset, {len(remaining)} still remaining")

    if not remaining:
        print("Nothing left to do — every id is already judged.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_subsets
    chunk_size = (len(remaining) + n - 1) // n
    for i in range(n):
        chunk = remaining[i * chunk_size: (i + 1) * chunk_size]
        if not chunk:
            break
        subset_path = out_dir / f"subset_{i + 1}.txt"
        subset_path.write_text("\n".join(chunk) + "\n")
        print(f"  subset_{i + 1}.txt: {len(chunk)} ids")

    print(f"\nUpload each subset_N.txt to its own Kaggle/Colab session, "
          f"set SUBSET_PATH accordingly in the notebook.")


if __name__ == "__main__":
    main()
