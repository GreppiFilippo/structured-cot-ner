#!/bin/bash
# Variants of the structured format of the free vs structured CoT experiment, on
# one node, one variant per run:
#
#   noedge       TYPES and MENTIONS, without EDGE
#   noedge_inv   MENTIONS before TYPES, without EDGE (the default)
#
#   1. prepare: the training file from the structured one (make_noedge_data.py):
#      same queries and answers, EDGE removed from the prompt and the reasoning,
#      and for noedge_inv the two remaining fields swapped in both
#   2. train Qwen3-4B-Thinking-2507 on it, with the same settings as the free
#      and structured students (train_qwen.py defaults, --max-seq-length 5120)
#   3. evaluate the adapter with the prompt of its training data on CrossNER and
#      the Pile-NER test split (outputs_paired/), then on the nine biomedical
#      sets (outputs_paired_bio/)
#
# The noedge student of the thesis was run on Modal instead (modal_app.py::noedge).
#
# From the repository root:
#   SMOKE=1 ./05_openvocab_sft/sbatch_paired_noedge.sh        # a few minutes of GPU
#   ./05_openvocab_sft/sbatch_paired_noedge.sh                # noedge_inv
#   VARIANT=noedge ./05_openvocab_sft/sbatch_paired_noedge.sh
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE="${NODE:-moro232}"
GPU="nvidia_geforce_rtx_3090:1"
LOG_DIR="$REPO/slurm_logs"
mkdir -p "$LOG_DIR"
RUN="$REPO/05_openvocab_sft/run_docker_paired.sh"
MODEL="Qwen/Qwen3-4B-Thinking-2507"
VARIANT="${VARIANT:-noedge_inv}"
case "$VARIANT" in
    noedge) INVERT=() ;;
    noedge_inv) INVERT=(--invert) ;;
    *) echo "VARIANT must be noedge or noedge_inv" >&2; exit 1 ;;
esac

if [ -n "$SMOKE" ]; then
    PREFIX="smoke"
    TRAIN_EXTRA=(--n-train-samples 40 --max-steps 3 --eval-steps 3)
    EVAL_EXTRA=(--limit 5)
else
    PREFIX="qwen3-4b-thinking"
    TRAIN_EXTRA=(); EVAL_EXTRA=()
fi
TAG="${PREFIX}_${VARIANT}"
NAME="${PREFIX}_sft_${VARIANT}"
DATA="data/pilener_paired_${VARIANT}_cot.jsonl"
ADAPTER="/workspace/05_openvocab_sft/work/${TAG}/adapter"
EVAL_DATA=(paired_eval/data/crossner_{ai,literature,music,politics,science}.json paired_eval/data/pilener_test.json)
SETS=(ncbi bc5cdr bc2gm bc4chemd AnatEM JNLPBA JNLPBA-Rare MedMentions-Rare biored)
BIO_DATA=("${SETS[@]/#/paired_eval/data_bio/}"); BIO_DATA=("${BIO_DATA[@]/%/.json}")

submit() {  # submit <label> <previous job or ""> <workdir> <command...>
    local label="$1" prev="$2"; shift 2
    sbatch --parsable -N 1 --gpus="$GPU" -w "$NODE" --export=ALL --chdir="$REPO" \
        ${prev:+--dependency=afterok:$prev} \
        --output="$LOG_DIR/${label}_%j.out" --error="$LOG_DIR/${label}_%j.err" "$RUN" "$@"
}

prep=$(submit "${TAG}_prepare" "" 05_openvocab_sft python3 make_noedge_data.py "${INVERT[@]}" \
    --structured data/pilener_paired_structured_cot.jsonl --out "$DATA")
train=$(submit "train_${TAG}" "$prep" 05_openvocab_sft python3 train_qwen.py --model-id "$MODEL" \
    --data-path "$DATA" --output-dir "work/${TAG}" --skip-push \
    --max-seq-length 5120 "${TRAIN_EXTRA[@]}")
ev=$(submit "eval_${NAME}" "$train" 04_evaluation python3 eval_paired.py --model-id "$MODEL" \
    --prompt "$VARIANT" --run-name "$NAME" --adapter "$ADAPTER" --data "${EVAL_DATA[@]}" "${EVAL_EXTRA[@]}")
bio=$(submit "bio_eval_${NAME}" "$ev" 04_evaluation python3 eval_paired.py --model-id "$MODEL" \
    --prompt "$VARIANT" --run-name "$NAME" --adapter "$ADAPTER" --output-dir outputs_paired_bio \
    --data "${BIO_DATA[@]}" "${EVAL_EXTRA[@]}")
echo "prepare $prep | train $train | eval CrossNER + Pile-NER $ev | eval bio $bio"
echo "Submitted to $NODE. Monitor with: squeue -u \$USER ; logs in $LOG_DIR"
