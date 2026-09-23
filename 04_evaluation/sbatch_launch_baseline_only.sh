#!/bin/bash
# Same as sbatch_launch_pretrain_only.sh, but for the BASELINE model
# (google/medgemma-1.5-4b-it, no LoRA adapter) instead of pretrain.
#
# NODE is hardcoded to moro232 below, matching the pretrain launcher — if
# faretra has freed up by the time you run this, change NODE to "faretra"
# to get up to 4x parallelism instead of running one at a time.

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

NODE="moro232"
LOG_DIR="$(pwd)/slurm_logs"
mkdir -p "$LOG_DIR"

for ds in "${DATASETS[@]}"; do
    job_label="${ds}_baseline"
    echo "Submitting: ${job_label} on ${NODE}"
    sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w "$NODE" --export=ALL \
        --output="${LOG_DIR}/${job_label}_%j.out" \
        --error="${LOG_DIR}/${job_label}_%j.err" \
        run_docker_eval.sh "baseline" "$ds"
done

echo ""
echo "Submitted ${#DATASETS[@]} baseline jobs to ${NODE}."
echo "Monitor with: squeue"
echo "Once both pretrain and baseline are done for all 9 datasets, aggregate with:"
echo "  python3 aggregate_results.py"
