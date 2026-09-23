#!/bin/bash
# Same pattern as the extractor's/jury's run_docker.sh — GPU from SLURM
# via $CUDA_VISIBLE_DEVICES, shared /llms cache so MedGemma's weights are
# downloaded once, not once per user.
#
# Args: model_type dataset_name (e.g. "baseline" "ncbi", or "pretrain" "bc5cdr")

PHYS_DIR="$(pwd)"  # run this script from the project root
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
    python3 /workspace/evaluate.py \
    --model-type "${1}" \
    --dataset-name "${2}" "${@:3}"
