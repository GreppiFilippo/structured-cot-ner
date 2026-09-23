#!/bin/bash
# Launches ONLY the 9 pretrain-model evaluation jobs (not baseline),
# pinned to moro232 instead of faretra — faretra's 4 GPUs are currently
# saturated by other users' long-running jobs (checked via squeue), while
# moro232 appeared completely free. moro232 has only ONE GPU though, so
# these 9 jobs will queue and run one at a time, not in parallel like they
# would on faretra's 4 GPUs — slower wall-clock time, but they start
# immediately instead of sitting in an already-long queue.
#
# Re-run this exact command later for the baseline jobs (swap "pretrain"
# for "baseline" below) once you're ready for the full comparison — or if
# faretra frees up in the meantime, that version can go back to using
# faretra's original sbatch_launch_eval.sh for more parallelism.

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
    job_label="${ds}_pretrain"
    echo "Submitting: ${job_label} on ${NODE}"
    sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w "$NODE" --export=ALL \
        --output="${LOG_DIR}/${job_label}_%j.out" \
        --error="${LOG_DIR}/${job_label}_%j.err" \
        run_docker_eval.sh "pretrain" "$ds"
done

echo ""
echo "Submitted ${#DATASETS[@]} pretrain jobs to ${NODE} (single GPU — will run sequentially)."
echo "Monitor with: squeue"
echo "If a job dies or hits a time limit, just re-run its own"
echo "  run_docker_eval.sh pretrain <dataset_name> sbatch command again —"
echo "  it skips automatically if outputs/<dataset>/post_sft/metrics.json already exists."
