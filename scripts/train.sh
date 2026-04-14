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
if [[ -z "$MODEL" || -z "$METHOD" ]]; then
    echo "Usage: bash train.sh --model MODEL --method METHOD"
    echo ""
    echo "Arguments:"
    echo "  --model   MODEL   Base model short name"
    echo "                    Choices: vicuna, mistral, llama2, qwen2.5, llama3.1, gemma"
    echo "  --method  METHOD  Training method"
    echo "                    Choices: cat, ercat"
    exit 1
fi

if [[ -z "${MODEL_IDS[$MODEL]+_}" ]]; then
    echo "Error: Unknown model '${MODEL}'"
    echo "Valid models: vicuna, mistral, llama2, qwen2.5, llama3.1, gemma"
    exit 1
fi

if [[ "$METHOD" != "cat" && "$METHOD" != "ercat" ]]; then
    echo "Error: Unknown method '${METHOD}'"
    echo "Valid methods: cat, ercat"
    exit 1
fi

# ========== Set Variables ==========
MODEL_ID="${MODEL_IDS[$MODEL]}"
DATACOLLATOR="${DATACOLLATORS[$MODEL]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"
CONFIG_DIR="${SCRIPT_DIR}/../configs/train"
SAVE_DIR="${SCRIPT_DIR}/../results/${MODEL}/${METHOD}"

# ========== Run Training ==========
cd "${SRC_DIR}"

echo "=========================================="
echo " Training"
echo "   Model:       ${MODEL} (${MODEL_ID})"
echo "   Method:      ${METHOD}"
echo "   Datacollator: ${DATACOLLATOR}"
echo "   Trainer cfg: ${CONFIG_DIR}/trainer-${METHOD}.yaml"
echo "   Save dir:    ${SAVE_DIR}"
echo "=========================================="

python train.py \
    --model-id "${MODEL_ID}" \
    --lora-cfg-path "${CONFIG_DIR}/lora.yaml" \
    --datacollator "${DATACOLLATOR}" \
    --dataset harmbench \
    --utilityset ultrachat200k \
    --trainer-cfg-path "${CONFIG_DIR}/trainer-${METHOD}.yaml" \
    --save-dir "${SAVE_DIR}" \
    --save-name train

echo "=========================================="
echo " Training complete. Results saved to:"
echo "   ${SAVE_DIR}"
echo "=========================================="
