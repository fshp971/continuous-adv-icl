#!/bin/bash
set -e

# ========== Model Lookup Table ==========
declare -A MODEL_IDS=(
    ["vicuna"]="lmsys/vicuna-7b-v1.5"
    ["mistral"]="mistralai/Mistral-7B-Instruct-v0.3"
    ["llama2"]="meta-llama/Llama-2-7b-chat-hf"
    ["qwen2.5"]="Qwen/Qwen2.5-7B-Instruct"
    ["llama3.1"]="meta-llama/Llama-3.1-8B-Instruct"
    ["gemma"]="google/gemma-1.1-2b-it"
)

declare -A DATACOLLATORS=(
    ["vicuna"]="vicuna-chat"
    ["mistral"]="llama2-chat"
    ["llama2"]="llama2-chat"
    ["qwen2.5"]="qwen2-chat"
    ["llama3.1"]="llama3-chat"
    ["gemma"]="gemma-chat"
)

declare -A DATASET_MAP=(
    ["advbench"]="advbench-first50"
    ["harmbench"]="harmbench-test50"
)

VALID_ATTACKS=("gcg-1" "beast-1" "gcq-1" "autodan-zhu-1" "pair-4" "deepinception-1")

# ========== Parse Arguments ==========
MODEL=""
METHOD=""
ATTACK=""
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --attack) ATTACK="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Error: Unknown argument '$1'"; exit 1 ;;
    esac
done

# ========== Validate Arguments ==========
if [[ -z "$MODEL" || -z "$ATTACK" || -z "$DATASET" ]]; then
    echo "Usage: bash eval_robustness.sh --model MODEL --attack ATTACK --dataset DATASET [--method METHOD]"
    echo ""
    echo "Arguments:"
    echo "  --model    MODEL    Base model short name"
    echo "                      Choices: vicuna, mistral, llama2, qwen2.5, llama3.1, gemma"
    echo "  --attack   ATTACK   Jailbreak attack type"
    echo "                      Choices: gcg-1, beast-1, gcq-1, autodan-zhu-1, pair-4, deepinception-1"
    echo "  --dataset  DATASET  Evaluation dataset"
    echo "                      Choices: advbench, harmbench"
    echo "  --method   METHOD   (Optional) Training method of the finetuned model to evaluate"
    echo "                      Choices: cat, ercat"
    echo "                      If omitted, evaluates the vanilla (pre-trained) base model"
    exit 1
fi

if [[ -z "${MODEL_IDS[$MODEL]+_}" ]]; then
    echo "Error: Unknown model '${MODEL}'"
    echo "Valid models: vicuna, mistral, llama2, qwen2.5, llama3.1, gemma"
    exit 1
fi

ATTACK_VALID=false
for a in "${VALID_ATTACKS[@]}"; do
    if [[ "$a" == "$ATTACK" ]]; then ATTACK_VALID=true; break; fi
done
if [[ "$ATTACK_VALID" == false ]]; then
    echo "Error: Unknown attack '${ATTACK}'"
    echo "Valid attacks: ${VALID_ATTACKS[*]}"
    exit 1
fi

if [[ -z "${DATASET_MAP[$DATASET]+_}" ]]; then
    echo "Error: Unknown dataset '${DATASET}'"
    echo "Valid datasets: advbench, harmbench"
    exit 1
fi

if [[ -n "$METHOD" && "$METHOD" != "cat" && "$METHOD" != "ercat" ]]; then
    echo "Error: Unknown method '${METHOD}'"
    echo "Valid methods: cat, ercat"
    exit 1
fi

# ========== Set Variables ==========
MODEL_ID="${MODEL_IDS[$MODEL]}"
DATACOLLATOR="${DATACOLLATORS[$MODEL]}"
DATASET_FULL="${DATASET_MAP[$DATASET]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
CONFIG_DIR="${SCRIPT_DIR}/../configs"

LORA_ARGS=""
if [[ -n "$METHOD" ]]; then
    TRAIN_DIR="${SCRIPT_DIR}/../results/${MODEL}/${METHOD}"
    SAVE_DIR="${TRAIN_DIR}/eval-${ATTACK}"
    LORA_ARGS="--lora-cfg-path ${CONFIG_DIR}/train/lora.yaml --model-resume-path ${TRAIN_DIR}/train_fin-model/adapter_model.safetensors"
    MODE_LABEL="finetuned (${METHOD})"
else
    SAVE_DIR="${SCRIPT_DIR}/../results/${MODEL}/vanilla/eval-${ATTACK}"
    MODE_LABEL="vanilla"
fi

# ========== Run Evaluation ==========
cd "${SRC_DIR}"

echo "=========================================="
echo " Robustness Evaluation"
echo "   Model:       ${MODEL} (${MODEL_ID})"
echo "   Mode:        ${MODE_LABEL}"
echo "   Attack:      ${ATTACK}"
echo "   Dataset:     ${DATASET} (${DATASET_FULL})"
echo "   Datacollator: ${DATACOLLATOR}"
echo "   Save dir:    ${SAVE_DIR}"
echo "=========================================="

# Step 1: Generate adversarial responses
echo "[Step 1/2] Generating adversarial responses..."

python evaluate.py \
    --model-id "${MODEL_ID}" \
    ${LORA_ARGS} \
    --dataset "${DATASET_FULL}" \
    --datacollator "${DATACOLLATOR}" \
    --evalset-cfg-path "${CONFIG_DIR}/eval/evalset.yaml" \
    --atker-cfg-path "${CONFIG_DIR}/eval/atk/${ATTACK}.yaml" \
    --save-dir "${SAVE_DIR}" \
    --exp-type build-evalset \
    --save-name "build-${DATASET}"

# Step 2: Judge responses (compute ASR)
echo "[Step 2/2] Judging generated responses (computing ASR)..."

python evaluate.py \
    --judger-cfg-path "${CONFIG_DIR}/eval/judge.yaml" \
    --evalset-path "${SAVE_DIR}/build-${DATASET}_evalset.json" \
    --save-dir "${SAVE_DIR}" \
    --exp-type judge-evalset \
    --save-name "judge-${DATASET}"

echo "=========================================="
echo " Robustness evaluation complete. Results saved to:"
echo "   ${SAVE_DIR}"
echo "=========================================="
