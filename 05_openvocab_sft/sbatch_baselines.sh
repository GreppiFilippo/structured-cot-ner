#!/bin/bash
# Published open NER models (eval_baselines.py: GLiNER-L v2.1, UniNER-7B-type)
# on the CrossNER sets (results in 04_evaluation/outputs_paired/) and on the 9
# biomedical sets (04_evaluation/outputs_paired_bio/), with the same queries and
# metric as the CoT students. Not on Pile-NER: both models were trained on
# Pile passages. Logs: slurm_logs/baseline_*. Image: paired-baselines-image.
#
# From the repository root (AFTER=<job id> chains the first job after it):
#   SMOKE=1 ./05_openvocab_sft/sbatch_baselines.sh   # 3 sentences of crossner_ai and ncbi
#   ./05_openvocab_sft/sbatch_baselines.sh

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE="${NODE:-moro232}"
GPU="nvidia_geforce_rtx_3090:1"
LOG_DIR="$REPO/slurm_logs"
mkdir -p "$LOG_DIR"
RUN="$REPO/05_openvocab_sft/run_docker_paired.sh"
export PAIRED_IMAGE=paired-baselines-image
CROSSNER=(paired_eval/data/crossner_{ai,literature,music,politics,science}.json)
BIO=(paired_eval/data_bio/{ncbi,bc5cdr,bc2gm,bc4chemd,AnatEM,JNLPBA,JNLPBA-Rare,MedMentions-Rare,biored}.json)
if [ -n "$SMOKE" ]; then
    PREFIX="smoke_"; CROSSNER=(paired_eval/data/crossner_ai.json); BIO=(paired_eval/data_bio/ncbi.json); EXTRA="--limit 3"
else
    PREFIX=""; EXTRA=""
fi

prev="${AFTER:-}"
for spec in "gliner gliner-l-v2.1" "uniner uniner-7b-type"; do
    set -- $spec
    model="$1"; name="${PREFIX}$2"
    cmd="python3 eval_baselines.py --model $model --run-name $name $EXTRA"
    prev=$(sbatch --parsable -N 1 --gpus="$GPU" -w "$NODE" --export=ALL --chdir="$REPO" \
        ${prev:+--dependency=afterany:$prev} \
        --output="$LOG_DIR/baseline_${name}_%j.out" --error="$LOG_DIR/baseline_${name}_%j.err" \
        "$RUN" 04_evaluation bash -c \
        "$cmd --output-dir outputs_paired --data ${CROSSNER[*]} && $cmd --output-dir outputs_paired_bio --data ${BIO[*]}")
    echo "$name: job $prev"
done
echo "last job: $prev"
