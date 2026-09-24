#!/bin/bash
# Same pattern as run_docker_eval.sh, but runs sanity_check.py instead of
# evaluate.py — meant for a quick one-off sbatch job (should take under a
# minute of actual GPU time), not a full evaluation run.
#
# Args: model_type [dataset_name] [extra sanity_check.py flags, e.g. --max-new-tokens 5120]
# (dataset_name defaults to "ncbi" if omitted)

PHYS_DIR="$(pwd)"
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

[ -f .env ] && export $(grep -v '^#' .env | xargs)

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    --rm \
    --memory="64g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    bioner-eval-image \
    python3 /workspace/sanity_check.py \
    --model-type "${1}" \
    --dataset-name "${2:-ncbi}" "${@:3}"
