#!/bin/bash
# 2026-08-14: switched from contiguous index-range shards to id-list
# subsets (see build_subsets.py) — a contiguous range silently never
# retries a row that failed earlier and was never completed, once a later
# shard's range starts past that row's position. Subsets computed from the
# exact remaining id set don't have that gap.
#
# Run build_subsets.py FIRST, pointing at whatever output file(s) already
# exist:
#   python3 build_subsets.py outputs/free_cot_0_45889.jsonl --n-subsets 4
# That writes outputs/subset_1.txt ... subset_N.txt. This script then
# submits one job per subset file found.
#
# Also splits across faretra's other GPUs — occupying a single GPU for ~10 days straight is not acceptable on this
# shared cluster. SLURM assigns each job whichever GPU is free on faretra
# when it starts, no need to pin GPU indices manually.
#
# Pinned to faretra with -w: that's where you built the Docker image, and
# the cluster has no shared filesystem — a job landing on a different node
# would fail immediately with "image/code not found", not a real error.

# Change to your own HF username/repo name — it gets auto-created (private,
# via create_repo(..., exist_ok=True) in src/hf_sync.py) if it doesn't exist
# yet, as long as HF_TOKEN in .env has write access. All shards push to the
# same repo, under their own distinct filenames — no collision.
export HF_BACKUP_REPO_ID="filippogreppi/free-cot-pile-ner"

LOG_DIR="$(pwd)/slurm_logs"
mkdir -p "$LOG_DIR"

shopt -s nullglob
subset_files=(outputs/subset_*.txt)
shopt -u nullglob

if [ ${#subset_files[@]} -eq 0 ]; then
    echo "No outputs/subset_*.txt files found — run build_subsets.py first, e.g.:"
    echo "  python3 build_subsets.py outputs/free_cot_0_45889.jsonl --n-subsets 4"
    exit 1
fi

for subset_path in "${subset_files[@]}"; do
    n_ids=$(wc -l < "$subset_path")
    echo "Submitting ${subset_path} (${n_ids} ids)"
    sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
        --output="${LOG_DIR}/$(basename "$subset_path" .txt)_%j.out" \
        --error="${LOG_DIR}/$(basename "$subset_path" .txt)_%j.err" \
        run_docker.sh "$subset_path"
done

echo ""
echo "Submitted ${#subset_files[@]} shard jobs. Monitor with: squeue"
echo "If a shard job dies or hits a time limit, just re-run its own"
echo "  run_docker.sh <subset_path> sbatch command again — it resumes from"
echo "  its own output file (ids already present are skipped automatically)."
echo "Once ALL shards (and any earlier files) are done, merge everything with:"
echo "  python3 merge_shards.py"
