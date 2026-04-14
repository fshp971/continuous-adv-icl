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

# ========== Parse Arguments ==========
MODEL=""
METHOD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        *) echo "Error: Unknown argument '$1'"; exit 1 ;;
    esac
done

# ========== Validate Arguments ==========
if [[ -z "$MODEL" ]]; then
    echo "Usage: bash eval_utility.sh --model MODEL [--method METHOD]"
    echo ""
    echo "Arguments:"
    echo "  --model   MODEL   Base model short name"
    echo "                    Choices: vicuna, mistral, llama2, qwen2.5, llama3.1, gemma"
    echo "  --method  METHOD  (Optional) Training method of the finetuned model to evaluate"
    echo "                    Choices: cat, ercat"
    echo "                    If omitted, evaluates the vanilla (pre-trained) base model"
    exit 1
fi

if [[ -z "${MODEL_IDS[$MODEL]+_}" ]]; then
    echo "Error: Unknown model '${MODEL}'"
    echo "Valid models: vicuna, mistral, llama2, qwen2.5, llama3.1, gemma"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
CONFIG_DIR="${SCRIPT_DIR}/../configs"

LORA_ARGS=""
if [[ -n "$METHOD" ]]; then
    TRAIN_DIR="${SCRIPT_DIR}/../results/${MODEL}/${METHOD}"
    SAVE_DIR="${TRAIN_DIR}/eval-alpacaeval"
    LORA_ARGS="--lora-cfg-path ${CONFIG_DIR}/train/lora.yaml --model-resume-path ${TRAIN_DIR}/train_fin-model/adapter_model.safetensors"
    MODE_LABEL="finetuned (${METHOD})"
else
    SAVE_DIR="${SCRIPT_DIR}/../results/${MODEL}/vanilla/eval-alpacaeval"
    MODE_LABEL="vanilla"
fi

# ========== Run Evaluation ==========
cd "${SRC_DIR}"

echo "=========================================="
echo " Utility Evaluation (AlpacaEval)"
echo "   Model:       ${MODEL} (${MODEL_ID})"
echo "   Mode:        ${MODE_LABEL}"
echo "   Datacollator: ${DATACOLLATOR}"
echo "   Save dir:    ${SAVE_DIR}"
echo "=========================================="

python evaluate.py \
    --model-id "${MODEL_ID}" \
    ${LORA_ARGS} \
    --datacollator "${DATACOLLATOR}" \
    --alpacaeval-cfg-path "${CONFIG_DIR}/eval/alpacaeval.yaml" \
    --save-dir "${SAVE_DIR}" \
    --exp-type build-alpacaeval \
    --save-name alpacaeval

echo "=========================================="
echo " Utility evaluation complete. Results saved to:"
echo "   ${SAVE_DIR}/alpacaeval_alpacaeval.json"
echo "=========================================="
