#!/bin/bash
# Submits 18 sbatch jobs — 9 datasets x 2 models (baseline, pretrain) —
# each on its own GPU, reading from the fixed subset on HF
# (structured-cot-ner/bioner-eval-subset) instead of re-deriving prompts from
# the raw IOB datasets each time.
#
# Unlike the original Kaggle notebook (which loaded both models
# sequentially on ONE GPU to fit a single session), each job here loads
# only the ONE model it needs — no reason to serialize when the cluster
# has multiple GPUs available.

DATASETS=(
    "ncbi"
    "bc5cdr"
    "bc2gm"
    "bc4chemd"
    "AnatEM"
    "JNLPBA"
    "JNLPBA-Rare"
    "MedMentions-Rare"
    "biored"
)

LOG_DIR="$(pwd)/slurm_logs"
mkdir -p "$LOG_DIR"

for ds in "${DATASETS[@]}"; do
    for model_type in "baseline" "pretrain"; do
        job_label="${ds}_${model_type}"
        echo "Submitting: ${job_label}"
        sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra --export=ALL \
            --output="${LOG_DIR}/${job_label}_%j.out" \
            --error="${LOG_DIR}/${job_label}_%j.err" \
            run_docker_eval.sh "$model_type" "$ds"
    done
done

echo ""
echo "Submitted ${#DATASETS[@]} datasets x 2 models = $((${#DATASETS[@]} * 2)) jobs."
echo "Monitor with: squeue"
echo "If a job dies or hits a time limit, just re-run its own"
echo "  run_docker_eval.sh <model_type> <dataset_name> sbatch command again —"
echo "  it skips automatically if outputs/<dataset>/<pre|post>_sft/metrics.json already exists."
echo "Once all 18 finish, aggregate with:"
echo "  python3 aggregate_results.py"
