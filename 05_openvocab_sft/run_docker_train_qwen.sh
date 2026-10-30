#!/bin/bash
# Cluster: LoRA SFT of a Qwen3 model (train_qwen.py) inside the existing
# cot-sft-image (built from ../03_sft_training/build/Dockerfile: torch
# 2.6+cu124, unpinned transformers/peft/datasets — nothing else needed).
# GPU from SLURM via $CUDA_VISIBLE_DEVICES, same pattern as the other stages.
#
# Args: model_id data_path tag hf_repo_id [extra train_qwen.py flags]
#   data_path: relative to this folder, or hf://<owner>/<dataset-repo>/<file>
#   tag:       names the work folder (work/<tag>), so runs never share checkpoints
# e.g. run_docker_train_qwen.sh Qwen/Qwen3-4B \
#          hf://structured-cot-ner/openvocab-sft/openvocab_train_v3_qwen.jsonl \
#          qwen3-4b_v3 filippogreppi/Qwen3-4B-openvocab-cot-v3
#
# HF_TOKEN (write access, to push the adapter) from .env here, or from
# ../03_sft_training/.env if this folder has none.

PHYS_DIR="$(pwd)"
LLM_CACHE_DIR="${HOME}/hf_cache"
DOCKER_INTERNAL_CACHE_DIR="/llms"
mkdir -p "$LLM_CACHE_DIR"

ENV_FILE=.env
[ -f "$ENV_FILE" ] || ENV_FILE=../03_sft_training/.env
[ -f "$ENV_FILE" ] && export $(grep -v '^#' "$ENV_FILE" | xargs)

case "${2}" in
    hf://*) DATA_PATH="${2}" ;;
    *) DATA_PATH="/workspace/${2}" ;;
esac

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e SFT_REPORT_TO="${SFT_REPORT_TO:-none}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-openvocab-cot-sft}" \
    --rm \
    --memory="64g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    cot-sft-image \
    python3 /workspace/train_qwen.py \
    --model-id "${1}" \
    --data-path "$DATA_PATH" \
    --output-dir "/workspace/work/${3}" \
    --hf-repo-id "${4}" \
    "${@:5}"
