#!/bin/bash
# Builds subset files PER JUDGE (each judge's own "remaining" set differs,
# since their completion progress differs — Granite/DeepSeek may already
# have some rows scored from earlier smoke tests, Gemma may have a few
# from the Colab test), then submits one sbatch job per (judge, subset)
# pair — N_SUBSETS x 3 judges total.
#
# Same architecture as the extractor's own subset-based sharding
# (build_subsets.py + sbatch_launch.sh) — one job per subset, applied
# here per-judge instead of per-GPU-on-one-node.
#
# NOTE on Gemma specifically: this submits Gemma jobs on vLLM same as the
# other two, but a real, isolated bug was found where vLLM returns empty
# content for Gemma specifically (confirmed: identical prompts work fine
# via plain transformers, fail every time via vLLM's OpenAI-compatible
# response packaging — see jury_gemma_production.ipynb for the Colab/
# Kaggle-based workaround). Splitting into smaller subsets does NOT fix
# that bug — expect Gemma's vLLM jobs here to still fail until that's
# separately resolved. Left in for when it eventually is, and because
# Granite/DeepSeek benefit right now regardless.

set -e

N_SUBSETS="${N_SUBSETS:-4}"
NODE="${NODE:-moro232}"
INPUT_PATH="${INPUT_PATH:-data/free_cot_merged.jsonl}"
OUTPUT_DIR="outputs"
LOG_DIR="slurm_logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

JUDGES=(
    "google/gemma-3-4b-it"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "ibm-granite/granite-4.1-8b"
)

for judge_model in "${JUDGES[@]}"; do
    judge_label=$(basename "$judge_model" | tr '.' '-')
    echo ""
    echo "=== ${judge_model} ==="

    # File gia' completati da QUESTO giudice (smoke test precedenti, run
    # parziali, ecc.) — ognuno ha il proprio progresso, quindi i
    # sottoinsiemi vanno calcolati separatamente per ciascuno, non un
    # unico split condiviso da tutti e tre.
    existing_outputs=("${OUTPUT_DIR}"/jury_scores_"${judge_label}"*.jsonl)
    # l'espansione del glob non trova nulla -> resta il pattern letterale,
    # non un file vero: filtriamo quelli che esistono davvero
    real_outputs=()
    for f in "${existing_outputs[@]}"; do
        [ -f "$f" ] && real_outputs+=("$f")
    done

    python3 subset_tools/build_judge_subsets.py \
        "$INPUT_PATH" "${real_outputs[@]}" \
        --n-subsets "$N_SUBSETS" \
        --output-dir "${OUTPUT_DIR}/subsets_${judge_label}"

    for i in $(seq 1 "$N_SUBSETS"); do
        subset_file="${OUTPUT_DIR}/subsets_${judge_label}/subset_${i}.txt"
        if [ ! -f "$subset_file" ]; then
            echo "  subset_${i}: niente da fare, salto (il giudice ha gia' finito o non ci sono abbastanza righe rimaste)"
            continue
        fi
        job_label="${judge_label}_subset${i}"
        echo "  Submitting: ${job_label} on ${NODE}"
        sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w "$NODE" --export=ALL \
            --output="${LOG_DIR}/${job_label}_%j.out" \
            --error="${LOG_DIR}/${job_label}_%j.err" \
            run_docker_judge.sh "$judge_model" "$INPUT_PATH" "" "$subset_file"
    done
done

echo ""
echo "Tutti i job sottomessi. Monitor con: squeue"
echo ""
echo "Quando tutti i job di UN giudice sono finiti, unisci i suoi pezzi:"
echo "  python3 subset_tools/merge_judge_parts.py outputs/jury_scores_<giudice>_subset_*.jsonl \\"
echo "      --output outputs/jury_scores_<giudice>.jsonl"
echo ""
echo "Poi, con tutti e tre i giudici uniti, aggrega come al solito:"
echo "  python3 aggregate_jury_scores.py outputs/jury_scores_*.jsonl"
