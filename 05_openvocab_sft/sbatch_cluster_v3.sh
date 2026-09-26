#!/bin/bash
# Submits every cluster job of the v3 open-vocabulary experiment at once:
#
#   1. train Qwen3-4B (bf16 LoRA, train_qwen.py) on the v3 dataset, <think> format
#   2. train MedGemma-1.5-4B (train_sft.py) on the same v3 dataset, in its own
#      <unused94>thought format, with the same epochs / LoRA alpha / learning
#      rate / effective batch as the Qwen run
#   3. evaluate Qwen3-4B base (zero-shot) on the 9 benchmarks — starts now
#   4. evaluate each fine-tuned adapter on the 9 benchmarks — each job waits
#      for its training job (--dependency=afterok) and pulls the adapter
#      from the Hub, where the training job pushed it
#
# All evaluations use the baseline's closed-schema prompt and the fixed
# subset, results in 04_evaluation/outputs/<dataset>/<run-name>/.
#
# Run from anywhere:  NODE=faretra ./05_openvocab_sft/sbatch_cluster_v3.sh
# Prerequisites: .env with a WRITE HF_TOKEN in 03_sft_training/ (training
# pushes the adapters) and in 04_evaluation/ (evaluation pulls them);
# images cot-sft-image and bioner-eval-image built on $NODE.

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE="${NODE:-faretra}"
GPU="nvidia_geforce_rtx_3090:1"
LOG_DIR="$REPO/slurm_logs"
mkdir -p "$LOG_DIR"

HF_DATA="hf://filippogreppi/openvocab-cot-sft"
QWEN_ID="Qwen/Qwen3-4B"
QWEN_ADAPTER="filippogreppi/Qwen3-4B-openvocab-cot-v3"
MEDGEMMA_ADAPTER="filippogreppi/MedGemma-openvocab-cot-v3"
DATASETS=(ncbi bc5cdr bc2gm bc4chemd AnatEM JNLPBA JNLPBA-Rare MedMentions-Rare biored)
# thinking models: generous budget (Qwen3-1.7B on the same data never
# exceeded 4096 tokens) and sampling as recommended by the model card
EVAL_TOKENS=6144

submit() {  # submit <label> <workdir> [sbatch options...] -- <script> [args...]
    local label="$1" dir="$2"; shift 2
    local opts=()
    while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    sbatch --parsable -N 1 --gpus="$GPU" -w "$NODE" --export=ALL --chdir="$dir" \
        --output="$LOG_DIR/${label}_%j.out" --error="$LOG_DIR/${label}_%j.err" "${opts[@]}" "$@"
}

jid_qwen=$(submit train_qwen3-4b_v3 "$REPO/05_openvocab_sft" -- \
    run_docker_train_qwen.sh "$QWEN_ID" "$HF_DATA/openvocab_train_v3_qwen.jsonl" qwen3-4b_v3 "$QWEN_ADAPTER")
echo "train Qwen3-4B: job $jid_qwen"

jid_medgemma=$(submit train_medgemma_v3 "$REPO/03_sft_training" -- \
    run_docker_sft.sh "$HF_DATA/openvocab_train_v3_medgemma.jsonl" "$MEDGEMMA_ADAPTER" \
    --work-dir /workspace/work_openvocab_v3 --num-epochs 2 --lora-alpha 32 --learning-rate 1e-4 \
    --per-device-batch 2 --grad-accum-steps 4)
echo "train MedGemma: job $jid_medgemma"

for ds in "${DATASETS[@]}"; do
    j1=$(submit "eval_qwen3-4b_base_$ds" "$REPO/04_evaluation" -- \
        run_docker_eval.sh causal "$ds" --model-id "$QWEN_ID" --run-name qwen3-4b_base \
        --max-new-tokens "$EVAL_TOKENS" --do-sample --batch-size 8)
    j2=$(submit "eval_qwen3-4b_sft_v3_$ds" "$REPO/04_evaluation" --dependency="afterok:$jid_qwen" -- \
        run_docker_eval.sh causal "$ds" --model-id "$QWEN_ID" --adapter "$QWEN_ADAPTER" \
        --run-name qwen3-4b_sft_v3 --max-new-tokens "$EVAL_TOKENS" --do-sample --batch-size 8)
    j3=$(submit "eval_medgemma_sft_v3_$ds" "$REPO/04_evaluation" --dependency="afterok:$jid_medgemma" -- \
        run_docker_eval.sh medgemma "$ds" --adapter "$MEDGEMMA_ADAPTER" --run-name medgemma_sft_v3 \
        --max-new-tokens "$EVAL_TOKENS" --batch-size 4)
    echo "$ds: base $j1 | Qwen sft $j2 (after $jid_qwen) | MedGemma sft $j3 (after $jid_medgemma)"
done

echo ""
echo "Submitted 2 trainings + 27 evaluations to $NODE. Monitor with: squeue -u \$USER"
echo "A failed or killed job can simply be resubmitted: training resumes from its last"
echo "checkpoint, evaluation from its last generated batch."
