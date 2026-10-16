#!/bin/bash
# Runs one command of the free vs structured CoT experiment inside
# tutor-cot-image (build_tutor/Dockerfile), same pattern as the other
# stages: GPU from SLURM via $CUDA_VISIBLE_DEVICES, HF cache in ~/hf_cache.
# The whole repository is mounted at /workspace, so training writes its
# adapters to 05_openvocab_sft/work/ and the evaluation reads them there.
#
# Args: <workdir relative to the repo root> <command...>
# e.g. run_docker_tutor.sh 04_evaluation python3 eval_tutor.py --prompt free ...
#
# TUTOR_IMAGE selects another image (e.g. tutor-baselines-image for eval_baselines.py).
# HF_TOKEN (read access to disi-unibo-nlp-students) from 05_openvocab_sft/.env,
# or from 03_sft_training/.env if that one does not exist.

# run from the repository root (sbatch_tutor.sh submits with --chdir): under SLURM, $0 is a copy
# of this script in the spool directory, so the repository cannot be derived from it
REPO="$(pwd)"
LLM_CACHE_DIR="${HOME}/hf_cache"
mkdir -p "$LLM_CACHE_DIR"

ENV_FILE="$REPO/05_openvocab_sft/.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$REPO/03_sft_training/.env"
[ -f "$ENV_FILE" ] && export $(grep -v '^#' "$ENV_FILE" | xargs)

WORKDIR="$1"; shift

# --ipc=host: vLLM and the PyTorch data loader need more shared memory than Docker's 64 MB default.
# Docker is rootless on the cluster nodes: root in the container is the user on the host, so the
# outputs are already owned by the user (a --user uid would instead map to an unprivileged one).
docker run \
    -v "$REPO":/workspace \
    -v "$LLM_CACHE_DIR":/llms \
    -e HF_HOME=/llms \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e PYTHONUNBUFFERED=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --ipc=host \
    --rm \
    --memory="28g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    -w "/workspace/$WORKDIR" \
    "${TUTOR_IMAGE:-tutor-cot-image}" \
    "$@"
