#!/bin/bash
# Starts a local vLLM OpenAI-compatible server for a judge model in bf16
# (no quantization — see build/requirements.txt for why), waits for it to
# become healthy, runs main_judge.py against the given extraction file,
# then always shuts the server down (even on failure) so the GPU is
# released cleanly.
#
# Args: judge_model input_path [limit] [id_subset_file]
# (e.g. "google/gemma-3-4b-it" "outputs/free_cot_merged.jsonl" 10, or
#  "google/gemma-3-4b-it" "outputs/free_cot_merged.jsonl" "" "outputs/subset_2.txt")
# limit is optional — pass it for a smoke test on just the first N rows,
# omit it (or leave the third arg empty) for a real full run.
# id_subset_file is optional — pass a subset_N.txt from
# build_judge_subsets.py to score only that subset (for sharding one
# judge across several parallel sbatch jobs, same idea as the extractor's
# own subset-based sharding); omit for scoring the whole input_path.
#
# 2026-08-29: sampling params and max-model-len made configurable via
# env vars (set before sbatch, propagated through --export=ALL) after
# checking each judge's own official model card — their recommended
# values genuinely differ, so a single hardcoded set was never right for
# more than one of the three:
#   Gemma-3-4b-it:          JUDGE_TEMPERATURE=1.0 JUDGE_TOP_P=0.95 JUDGE_TOP_K=64
#   DeepSeek-R1-Distill-8B: JUDGE_TEMPERATURE=0.6 JUDGE_TOP_P=0.95
#                           JUDGE_MAX_TOKENS=8192 VLLM_MAX_MODEL_LEN=12288
#                           (their own card states 32,768 tokens as a
#                           normal generation length for this family —
#                           our previous 1024 default was 32x too small,
#                           the direct cause of "no parseable JSON, ran
#                           out of tokens mid-reasoning" failures seen in
#                           testing; 8192 is a practical middle ground,
#                           not the full 32,768, since a judge call
#                           happening thousands of times shouldn't pay
#                           for the rare worst case every time — and
#                           max-model-len must be raised too, since it
#                           caps prompt+generation together regardless of
#                           what max_tokens requests)
#   Granite-4.1-8B:         JUDGE_TEMPERATURE=1.0 JUDGE_TOP_P=0.95
# All defaults below are otherwise safe no-ops (Config's own defaults)
# when left unset.

set -e

[ -f .env ] && export $(grep -v '^#' .env | xargs)

judge_model="${1}"
input_path="${2}"
limit="${3:-}"
id_subset_file="${4:-}"
PORT=8000

# 2026-08-29: raised default from 4096 — earlier real logs (Granite)
# showed 28,064 tokens of KV cache available even at max-model-len=4096,
# confirming plenty of headroom on these smaller (4-8B, unquantized)
# models to raise this without a real memory risk, unlike the tight
# situation Qwen3-32B-AWQ was in.
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# 2026-08-27: log filename made judge-specific. All three judge jobs mount
# the SAME jury/ directory as /workspace (that's how the shared code and
# HF cache get in) — with the old fixed "vllm_server.log" name, running
# all three in parallel meant whichever finished writing last silently
# overwrote the other two, making it impossible to tell whose log you were
# actually looking at (this happened for real: a Granite log was mistaken
# for Gemma's).
#
# 2026-08-31: same reasoning extended to id_subset_file — running the SAME
# judge on 4 subsets in parallel (4 sbatch jobs) would otherwise collide
# on this same log file too.
judge_label=$(basename "$judge_model" | tr '.' '-')
if [ -n "$id_subset_file" ]; then
    subset_label=$(basename "$id_subset_file" .txt)
    VLLM_LOG="vllm_server_${judge_label}_${subset_label}.log"
else
    VLLM_LOG="vllm_server_${judge_label}.log"
fi

# No --reasoning-parser, same reasoning as the extractor's train.sh: none
# of the three judges are Qwen-family (chosen specifically to avoid
# same-family judge bias), so there's no shared reasoning-parser to rely
# on anyway. build_jury_prompt already asks for a plain JSON response, no
# <think> block expected from these judges under normal operation.
python3 -m vllm.entrypoints.openai.api_server \
    --model "$judge_model" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --port "$PORT" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Shutting down vLLM server (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to become healthy..."
until curl -sf "http://localhost:${PORT}/health" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server process died during startup, check ${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done
echo "vLLM server ready."

LIMIT_ARGS=""
if [ -n "$limit" ]; then
    LIMIT_ARGS="--limit ${limit}"
fi

SAMPLING_ARGS="--temperature ${JUDGE_TEMPERATURE:-0.3} --max-tokens ${JUDGE_MAX_TOKENS:-1024}"
if [ -n "${JUDGE_TOP_P:-}" ]; then
    SAMPLING_ARGS="$SAMPLING_ARGS --top-p ${JUDGE_TOP_P}"
fi
if [ -n "${JUDGE_TOP_K:-}" ]; then
    SAMPLING_ARGS="$SAMPLING_ARGS --top-k ${JUDGE_TOP_K}"
fi

SUBSET_ARGS=""
if [ -n "$id_subset_file" ]; then
    SUBSET_ARGS="--id-subset-file ${id_subset_file}"
fi

python3 main_judge.py \
    --judge-model "$judge_model" \
    --input-path "$input_path" \
    $LIMIT_ARGS \
    $SUBSET_ARGS \
    $SAMPLING_ARGS \
    --base-url "http://localhost:${PORT}/v1" \
    --hf-backup-repo-id "${HF_BACKUP_REPO_ID:-}"
