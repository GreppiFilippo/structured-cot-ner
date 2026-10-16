#!/bin/bash
# Completes the evaluation of the base model with the structured prompt on
# CrossNER and the Pile-NER test split: the first 100 sentences of each
# CrossNER domain and 200 of Pile-NER were evaluated as
# qwen3-4b-thinking_base_structured_sample; this job evaluates the rest with
# the same settings (eval_tutor.py --start) and joins the two parts into
# qwen3-4b-thinking_base_structured (merge_runs.py). Logs: slurm_logs/bstruct_*.
#
# From the repository root (AFTER=<job id> chains it after another job):
#   ./05_openvocab_sft/sbatch_tutor_base_structured.sh

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE="${NODE:-moro232}"
LOG_DIR="$REPO/slurm_logs"
mkdir -p "$LOG_DIR"
P="qwen3-4b-thinking"
CROSSNER="tutor_eval/data/crossner_ai.json tutor_eval/data/crossner_literature.json tutor_eval/data/crossner_music.json tutor_eval/data/crossner_politics.json tutor_eval/data/crossner_science.json"
PILENER="tutor_eval/data/pilener_test.json"
EVAL="python3 eval_tutor.py --model-id Qwen/Qwen3-4B-Thinking-2507 --prompt structured --run-name ${P}_base_structured_rest"
MERGE="python3 merge_runs.py --dir outputs_tutor --parts ${P}_base_structured_sample ${P}_base_structured_rest --out ${P}_base_structured"
job=$(sbatch --parsable -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w "$NODE" --export=ALL --chdir="$REPO" \
    ${AFTER:+--dependency=afterany:$AFTER} \
    --output="$LOG_DIR/bstruct_%j.out" --error="$LOG_DIR/bstruct_%j.err" \
    "$REPO/05_openvocab_sft/run_docker_tutor.sh" 04_evaluation bash -c \
    "$EVAL --start 100 --data $CROSSNER && $EVAL --start 200 --data $PILENER && $MERGE --data $CROSSNER $PILENER")
echo "base structured (rest + merge): job $job"
echo "last job: $job"
