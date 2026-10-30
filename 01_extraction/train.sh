#!/bin/bash
# Runs inside the container. Args: start_idx end_idx
#
# Starts a local vLLM OpenAI-compatible server for Qwen3-32B-AWQ (official
# pre-quantized 4-bit checkpoint, fits a single 24GB GPU), waits for it to
# become healthy, runs the extraction shard against it, then always shuts
# the server down (even on failure) so the GPU is released cleanly for the
# next job.

set -e

# Make .env variables (e.g. HF_TOKEN, if the model repo is gated) visible to
# the vllm subprocess too, not just to main.py's own dotenv loading.
[ -f .env ] && export $(grep -v '^#' .env | xargs)

start_idx="${1}"
end_idx="${2}"
MODEL="${MODEL_NAME:-Qwen/Qwen3-32B-AWQ}"
PORT=8000

# 2026-08-14: run_docker.sh's first argument can now be EITHER a start
# index (paired with an end index as the second argument, the original
# contiguous-range sharding) OR the path to a subset_N.txt from
# build_subsets.py (id-list-based sharding, second argument unused) — the
# id-list mode correctly retries rows that failed earlier anywhere in the
# dataset, which a contiguous range starting past that point never does.
# Detected here by file existence rather than a separate flag, so
# run_docker.sh itself needed no changes.
if [ -f "$start_idx" ]; then
    SUBSET_ARGS="--id-subset-file ${start_idx}"
    IDX_ARGS=""
else
    SUBSET_ARGS=""
    IDX_ARGS="--start-idx ${start_idx} --end-idx ${end_idx}"
fi

# Not using --reasoning-parser: testing against the real HF router + nscale
# backend showed inconsistent behavior (the JSON fence in `content` was
# present in some calls and absent in others, and disabling thinking via
# chat_template_kwargs was not respected). Capturing the full raw generation
# and splitting client-side (see src/extractor.py) is more robust and
# matches what already works in the Colab/Kaggle notebook.
#
# 2026-08-13: switched from Qwen3-32B + on-the-fly bitsandbytes quantization
# to Qwen/Qwen3-32B-AWQ, the official pre-quantized checkpoint Qwen
# themselves publish — on-the-fly bitsandbytes
# quantization is known to be slower (vLLM's own startup log literally
# warns "bitsandbytes quantization is not fully optimized yet"), and AWQ
# has dedicated, well-optimized kernels built into vLLM (Marlin/Triton,
# auto-selected — confirmed no extra package needed to *use* an
# already-quantized AWQ checkpoint, only to *create* one). No
# --quantization/--load-format flags needed: vLLM auto-detects AWQ from
# the model's own config.json. This also means the KV cache memory math
# in the comments below (calibrated for bitsandbytes' ~18 GiB weight
# footprint) is now stale — re-derive it from this model's own real
# startup log instead of assuming the old numbers still apply.
#
# 2026-08-04: max-model-len set to 4096 for the bitsandbytes setup, after
# three real failures at higher values there (kept as historical context
# for the math pattern — the actual ceiling itself needs re-measuring for
# AWQ's different memory footprint, not assumed to carry over unchanged).
# 2026-08-05: --enforce-eager tried and reverted under the old bitsandbytes
# setup — GPU KV cache size was 4,800 tokens vs 4,864 without eager mode,
# essentially unchanged, while generation throughput dropped 3-4x. Whether
# this still holds under AWQ hasn't been retested; treat as unverified
# under the new setup rather than settled.
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --port "$PORT" \
    > vllm_server.log 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Shutting down vLLM server (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to become healthy..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server process died during startup, check vllm_server.log"
        exit 1
    fi
    sleep 5
done
echo "vLLM server ready."

python3 main.py \
    $IDX_ARGS \
    $SUBSET_ARGS \
    --model-name "$MODEL" \
    --base-url "http://localhost:${PORT}/v1" \
    --hf-backup-repo-id "${HF_BACKUP_REPO_ID:-}"
