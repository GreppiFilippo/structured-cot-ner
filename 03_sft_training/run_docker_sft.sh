#!/bin/bash
# Same pattern as jury/run_docker_judge.sh and bioner_eval/run_docker_eval.sh.
#
# IMPORTANT: every env var the training script might need is explicitly
# forwarded via -e below. A real bug was found in run_docker_judge.sh
# (2026-09-14): a container does NOT automatically inherit the host's
# environment variables, only whatever is explicitly passed via -e — a
# missing -e for judge-specific sampling params caused hours of silent
# failures with DeepSeek, traced only after checking the actual vLLM
# server startup log. Every var this script's train_sft.py reads from
# os.environ (HF_TOKEN, SFT_REPORT_TO, WANDB_API_KEY if used) is listed
# explicitly here from the start, not added reactively after a failure.

PHYS_DIR="$(pwd)"
LLM_CACHE_DIR="${HOME}/hf_cache"
DOCKER_INTERNAL_CACHE_DIR="/llms"
mkdir -p "$LLM_CACHE_DIR"

[ -f .env ] && export $(grep -v '^#' .env | xargs)

# $1: a path relative to this folder, or hf://<owner>/<dataset-repo>/<file>
# (downloaded inside the container by train_sft.py). Extra arguments after
# $2 are passed to train_sft.py and override the defaults above, e.g.
# --work-dir /workspace/work_<run> so a new run never resumes the
# checkpoints of a previous one.
case "${1}" in
    hf://*) DATA_PATH="${1}" ;;
    *) DATA_PATH="/workspace/${1}" ;;
esac

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e SFT_REPORT_TO="${SFT_REPORT_TO:-none}" \
    -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
    -e WANDB_PROJECT="${WANDB_PROJECT:-medgemma-cot-sft}" \
    --rm \
    --memory="64g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    cot-sft-image \
    python3 /workspace/train_sft.py \
    --data-path "$DATA_PATH" \
    --hf-repo-id "${2:-filippogreppi/MedGemma-CoT-distilled}" \
    --work-dir /workspace/work \
    "${@:3}"
