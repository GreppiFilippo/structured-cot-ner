#!/bin/bash
# Follow-up experiments of the free vs structured CoT comparison, on the
# CrossNER sets and the Pile-NER test split (results next to the main runs,
# in 04_evaluation/outputs_paired/<dataset>/<run>/; logs slurm_logs/extra_*):
#
#   E1  structured student, constrained by the grammar of the structured run (three
#       one-line fields, then a generic {type: [strings]} JSON)
#   E2  structured student, same fields, answer keys fixed to the requested types
#   E3  base model, structured prompt, field grammar (no training: the
#       inference-only method the structured format comes from); E3b the same
#       with the answer keys fixed
#   E4  free student with each query's reasoning capped at the tokens the
#       structured student used on it (budget forcing)
#   E5  both students again with seed 1, CrossNER only (seed variance)
#
# Jobs are chained (each starts when the previous one ends, even if it failed).
# From the repository root:
#   SMOKE=1 ./05_openvocab_sft/sbatch_paired_extra.sh   # 3 sentences of crossner_ai per run
#   ./05_openvocab_sft/sbatch_paired_extra.sh

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE="${NODE:-moro232}"
GPU="nvidia_geforce_rtx_3090:1"
LOG_DIR="$REPO/slurm_logs"
mkdir -p "$LOG_DIR"
RUN="$REPO/05_openvocab_sft/run_docker_paired.sh"
MODEL="Qwen/Qwen3-4B-Thinking-2507"
P="qwen3-4b-thinking"
A_STRUCT="/workspace/05_openvocab_sft/work/${P}_structured/adapter"
A_FREE="/workspace/05_openvocab_sft/work/${P}_free/adapter"
CROSSNER=(paired_eval/data/crossner_{ai,literature,music,politics,science}.json)
ALL=("${CROSSNER[@]}" paired_eval/data/pilener_test.json)

if [ -n "$SMOKE" ]; then
    PREFIX="smoke_"; ALL=(paired_eval/data/crossner_ai.json); CROSSNER=("${ALL[@]}"); EXTRA=(--limit 3)
else
    PREFIX=""; EXTRA=()
fi

prev="${AFTER:-}"
submit() {  # submit <run name> <prompt> <data...> -- [extra eval_paired.py flags...]
    local name="${PREFIX}$1" prompt="$2"; shift 2
    local data=()
    while [ "$1" != "--" ]; do data+=("$1"); shift; done; shift
    prev=$(sbatch --parsable -N 1 --gpus="$GPU" -w "$NODE" --export=ALL --chdir="$REPO" \
        ${prev:+--dependency=afterany:$prev} \
        --output="$LOG_DIR/extra_${name}_%j.out" --error="$LOG_DIR/extra_${name}_%j.err" \
        "$RUN" 04_evaluation python3 eval_paired.py --model-id "$MODEL" --prompt "$prompt" --run-name "$name" \
        --data "${data[@]}" "${EXTRA[@]}" "$@")
    echo "$name: job $prev"
}

submit "${P}_sft_structured_grammar" structured "${ALL[@]}" -- --adapter "$A_STRUCT" --grammar fields
submit "${P}_sft_structured_grammar_keys" structured "${ALL[@]}" -- --adapter "$A_STRUCT" --grammar keys
submit "${P}_base_structured_grammar" structured "${ALL[@]}" -- --grammar fields
submit "${P}_base_structured_grammar_keys" structured "${ALL[@]}" -- --grammar keys
submit "${P}_sft_free_budget_structured" free "${ALL[@]}" -- --adapter "$A_FREE" \
    --think-budget-from "${P}_sft_structured"
if [ -z "$SMOKE" ]; then
    submit "${P}_sft_structured_seed1" structured "${CROSSNER[@]}" -- --adapter "$A_STRUCT" --seed 1
    submit "${P}_sft_free_seed1" free "${CROSSNER[@]}" -- --adapter "$A_FREE" --seed 1
fi
echo "last job: $prev"
