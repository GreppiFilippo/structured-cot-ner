#!/bin/bash
# Same pattern as the extractor's run_docker.sh — GPU from SLURM via
# $CUDA_VISIBLE_DEVICES.
#
# 2026-08-30: cache moved from the shared /llms to $HOME/hf_cache after a
# real disk-space crash on moro232 — /llms is shared across every user of
# the cluster (confirmed: an 8.1GB gemma-3-4b-it copy already sitting
# there belongs to a different user, "patrignani", not us), and it had
# under 3.6GB free while a ~5GB Gemma download needed more room. Using a
# private, per-user cache avoids depending on how much room other users'
# downloads happen to be leaving free at any given moment.
#
# 2026-09-01: briefly switched to --gpus all after a real failure running
# 4 subset jobs of the same judge in parallel on one node (all four failed
# identically with "CUDA unknown error ... Setting the available devices
# to be zero"), on the theory that SLURM's cgroup GPU isolation presents
# each job's assigned GPU as index 0 *relative to that job's restricted
# view*, and passing that same "0" straight into Docker's --gpus device=X
# would ask for physical GPU 0 literally.
#
# 2026-09-03: reverted back to device=$CUDA_VISIBLE_DEVICES per the
# cluster's own guide (http://faretra:37339/slurm/guide, section 6) —
# it states plainly that $CUDA_VISIBLE_DEVICES holds the *real* physical
# index SLURM assigned this job, node-relative, not a job-relative "0",
# and that this is the canonical pattern every project on this cluster
# is expected to use. The 2026-09-01 failure was real, but the fix above
# was the wrong theory for it — worth re-diagnosing separately if 4
# parallel subset jobs on one node are attempted again; for the normal
# single-job case this is what the cluster admins say to use.
PHYS_DIR="$(pwd)"  # run this script from the project root
LLM_CACHE_DIR="${HOME}/hf_cache"
DOCKER_INTERNAL_CACHE_DIR="/llms"
mkdir -p "$LLM_CACHE_DIR"

# 2026-09-14: JUDGE_TEMPERATURE/JUDGE_TOP_P/JUDGE_TOP_K/JUDGE_MAX_TOKENS/
# VLLM_MAX_MODEL_LEN added to the -e flags below after a real bug: even
# with these vars correctly set on the host (confirmed via sbatch
# --export=ALL and verified present in the job's own environment),
# train_judge.sh runs INSIDE this Docker container — and a container
# does not automatically inherit the host shell's environment variables,
# only whatever is explicitly passed via -e. These five were never
# forwarded, so train_judge.sh silently fell back to its own defaults
# (temperature=0.3, max_tokens=1024, max_model_len=8192) every single
# time, regardless of what was set before sbatch. Confirmed directly:
# a DeepSeek run launched with VLLM_MAX_MODEL_LEN=12288 still started
# its vLLM server with max_model_len=8192 (visible in the server's own
# startup log), causing the exact "ran out of tokens mid-reasoning,
# no parseable JSON" failures this fix addresses — on about half of
# the remaining rows in one real run.
docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_BACKUP_REPO_ID="${HF_BACKUP_REPO_ID:-}" \
    -e JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-}" \
    -e JUDGE_TOP_P="${JUDGE_TOP_P:-}" \
    -e JUDGE_TOP_K="${JUDGE_TOP_K:-}" \
    -e JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-}" \
    -e VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}" \
    --rm \
    --memory="64g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    pilener-jury-image \
    "/workspace/train_judge.sh" \
    "${1}" "${2}" "${3}" "${4}"