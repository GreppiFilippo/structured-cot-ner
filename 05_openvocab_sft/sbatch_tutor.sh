#!/bin/bash
# Submits the free vs structured CoT experiment (tutors' dataset
# disi-unibo-nlp-students/pilener-teacher-cot-agreement-v1) to one node:
#
#   1. prepare: the two training files (build_tutor_cot_dataset.py) and the
#      evaluation sets (build_tutor_eval.py: CrossNER, gold types + 1-3
#      distractors, seed 42; Pile-NER test split)
#   2. train Qwen3-4B-Thinking-2507 on structured_cot, then on free_cot
#      (LoRA, train_qwen.py defaults; every example kept: --max-seq-length 5120)
#   3. evaluate the base model with both prompts, and each adapter with the
#      prompt it was trained on (eval_tutor.py, vLLM)
#
# The node has one GPU, so the jobs run one after the other; dependencies
# only make each evaluation of an adapter wait for its training.
# Results: 04_evaluation/outputs_tutor/<dataset>/<run>/; adapters:
# 05_openvocab_sft/work/<run>/adapter.
#
# From the repository root, after building the image on the node:
#   SMOKE=1 ./05_openvocab_sft/sbatch_tutor.sh      # end-to-end check, a few minutes of GPU
#   ./05_openvocab_sft/sbatch_tutor.sh              # the experiment
# A failed or killed job can be resubmitted alone: training resumes from its
# last checkpoint, evaluation skips the datasets it already finished.

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE="${NODE:-moro232}"
GPU="nvidia_geforce_rtx_3090:1"
LOG_DIR="$REPO/slurm_logs"
mkdir -p "$LOG_DIR"
RUN="$REPO/05_openvocab_sft/run_docker_tutor.sh"
MODEL="Qwen/Qwen3-4B-Thinking-2507"

if [ -n "$SMOKE" ]; then
    PREFIX="smoke"; VARIANTS=(structured)
    TRAIN_EXTRA=(--n-train-samples 40 --max-steps 3 --eval-steps 3)
    EVAL_EXTRA=(--limit 5)
else
    PREFIX="qwen3-4b-thinking"; VARIANTS=(structured free)
    TRAIN_EXTRA=(); EVAL_EXTRA=()
fi

submit() {  # submit <label> [sbatch options...] -- <command...>
    local label="$1"; shift
    local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    sbatch --parsable -N 1 --gpus="$GPU" -w "$NODE" --export=ALL --chdir="$REPO" \
        --output="$LOG_DIR/${label}_%j.out" --error="$LOG_DIR/${label}_%j.err" "${opts[@]}" "$@"
}

prep=$(submit "${PREFIX}_prepare" -- "$RUN" . bash -c \
    "cd 05_openvocab_sft && python3 build_tutor_cot_dataset.py --out-dir data && \
     cd ../04_evaluation && python3 build_tutor_eval.py --out-dir tutor_eval/data")
echo "prepare: job $prep"

EVAL_DATA=(tutor_eval/data/crossner_{ai,literature,music,politics,science}.json tutor_eval/data/pilener_test.json)
evaluate() {  # evaluate <run name> <prompt> <dependency> [extra eval_tutor.py flags...]
    local name="$1" prompt="$2" dep="$3"; shift 3
    submit "eval_${name}" --dependency="afterok:$dep" -- "$RUN" 04_evaluation \
        python3 eval_tutor.py --model-id "$MODEL" --prompt "$prompt" --run-name "$name" \
        --data "${EVAL_DATA[@]}" "${EVAL_EXTRA[@]}" "$@"
}

for v in "${VARIANTS[@]}"; do
    tag="${PREFIX}_${v}"
    jt=$(submit "train_${tag}" --dependency="afterok:$prep" -- "$RUN" 05_openvocab_sft \
        python3 train_qwen.py --model-id "$MODEL" --data-path "data/pilener_tutor_${v}_cot.jsonl" \
        --output-dir "work/${tag}" --skip-push --max-seq-length 5120 "${TRAIN_EXTRA[@]}")
    je=$(evaluate "${PREFIX}_sft_${v}" "$v" "$jt" --adapter "/workspace/05_openvocab_sft/work/${tag}/adapter")
    jb=$(evaluate "${PREFIX}_base_${v}" "$v" "$prep")
    echo "$v: train $jt | eval sft $je (after $jt) | eval base $jb"
done

echo ""
echo "Submitted to $NODE. Monitor with: squeue -u \$USER ; logs in $LOG_DIR"
