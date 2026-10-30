#!/bin/bash
# Submits one sbatch job per judge — all three run in parallel on
# faretra's other GPUs, each scoring the SAME input file independently.
# Deliberately not Qwen-family (avoids same-family judge bias, per the
# design of the project — see jury_colab.ipynb).
#
# Usage: ./sbatch_launch_judge.sh outputs/free_cot_merged.jsonl

INPUT_PATH="${1:?Usage: ./sbatch_launch_judge.sh <path-to-extraction-jsonl>}"

JUDGES=(
    "google/gemma-3-4b-it"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "ibm-granite/granite-4.1-8b"
)

# Change to your own HF username/repo name — distinct from the extraction
# repo, since this holds a different kind of data (scores, not CoTs).
export HF_BACKUP_REPO_ID="filippogreppi/jury-scores"

LOG_DIR="$(pwd)/slurm_logs"
mkdir -p "$LOG_DIR"

for judge in "${JUDGES[@]}"; do
    judge_label=$(basename "$judge" | tr '.' '-')
    echo "Submitting judge: ${judge}"
    sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
        --output="${LOG_DIR}/judge_${judge_label}_%j.out" \
        --error="${LOG_DIR}/judge_${judge_label}_%j.err" \
        run_docker_judge.sh "$judge" "$INPUT_PATH"
done

echo ""
echo "Submitted ${#JUDGES[@]} judge jobs, all scoring: ${INPUT_PATH}"
echo "Monitor with: squeue"
echo "If a judge job dies or hits a time limit, just re-run its own"
echo "  run_docker_judge.sh <judge_model> <input_path> sbatch command again —"
echo "  it resumes from its own output file (rows already scored are skipped)."
echo "Once all three finish, aggregate with:"
echo "  python3 aggregate_jury_scores.py outputs/jury_scores_*.jsonl"
