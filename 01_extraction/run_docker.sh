#!/bin/bash
# Local inference now (Qwen3-32B via vLLM): needs the GPU SLURM assigned
# via $CUDA_VISIBLE_DEVICES, and the cluster's shared /llms cache so the 54GB
# checkpoint is downloaded once, not once per user (see the guide, Section 6).

PHYS_DIR="$(pwd)"  # run this script from the project root
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_BACKUP_REPO_ID="${HF_BACKUP_REPO_ID:-}" \
    --rm \
    --memory="64g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    pilener-cot-image \
    "/workspace/train.sh" \
    "${1}" "${2}"
