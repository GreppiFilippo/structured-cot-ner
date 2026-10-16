#!/bin/bash
# Biomedical evaluation of the free vs structured CoT experiment, kept
# apart from the CrossNER one (sbatch_tutor.sh): sets in
# 04_evaluation/tutor_eval/data_bio/ (build_tutor_eval_bio.py), results in
# 04_evaluation/outputs_tutor_bio/<dataset>/<run>/, logs slurm_logs/bio_*.
# Uses the two students already trained by sbatch_tutor.sh.
#
# Jobs, chained so that they run in this order on the node's one GPU (each
# starts when the previous one ends, even if it failed):
#   1. prepare the 9 sets (skipped if they already exist)
#   2. structured student, structured prompt
#   3. free student, free prompt
#   4. base model, free prompt
#   5. base model, structured prompt, on the first 50 sentences of each set
#      (enough to document whether it answers in the requested format)
#
# From the repository root:  ./05_openvocab_sft/sbatch_tutor_bio.sh

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE="${NODE:-moro232}"
GPU="nvidia_geforce_rtx_3090:1"
LOG_DIR="$REPO/slurm_logs"
mkdir -p "$LOG_DIR"
RUN="$REPO/05_openvocab_sft/run_docker_tutor.sh"
MODEL="Qwen/Qwen3-4B-Thinking-2507"
P="qwen3-4b-thinking"
OUT="outputs_tutor_bio"
SETS=(ncbi bc5cdr bc2gm bc4chemd AnatEM JNLPBA JNLPBA-Rare MedMentions-Rare biored)
DATA=("${SETS[@]/#/tutor_eval/data_bio/}"); DATA=("${DATA[@]/%/.json}")

submit() {  # submit <label> <previous job or ""> <command...>
    local label="$1" prev="$2"; shift 2
    sbatch --parsable -N 1 --gpus="$GPU" -w "$NODE" --export=ALL --chdir="$REPO" \
        ${prev:+--dependency=afterany:$prev} \
        --output="$LOG_DIR/bio_${label}_%j.out" --error="$LOG_DIR/bio_${label}_%j.err" "$RUN" "$@"
}

prev=""
missing=0
for f in "${DATA[@]}"; do [ -f "$REPO/04_evaluation/$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
    prev=$(submit prepare "" 04_evaluation python3 build_tutor_eval_bio.py --out-dir tutor_eval/data_bio)
    echo "prepare: job $prev"
else
    echo "prepare: sets already built, skipped"
fi

evaluate() {  # evaluate <run name> <prompt> [extra eval_tutor.py flags...]
    local name="$1" prompt="$2"; shift 2
    prev=$(submit "eval_${name}" "$prev" 04_evaluation python3 eval_tutor.py --model-id "$MODEL" \
        --prompt "$prompt" --run-name "$name" --output-dir "$OUT" --data "${DATA[@]}" "$@")
    echo "$name ($prompt prompt): job $prev"
}

evaluate "${P}_sft_structured" structured --adapter "/workspace/05_openvocab_sft/work/${P}_structured/adapter"
evaluate "${P}_sft_free" free --adapter "/workspace/05_openvocab_sft/work/${P}_free/adapter"
evaluate "${P}_base_free" free
evaluate "${P}_base_structured_sample" structured --limit 50

echo ""
echo "Submitted to $NODE, one after the other. Monitor with: squeue -u \$USER"
