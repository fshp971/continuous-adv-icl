# Tutorial on Running Scripts

[Scripts](./) are organized into three categories:

- [scripts/train.sh](./train.sh) — Adversarial training with the **CAT** or **ERCAT** method.
- [scripts/eval_robustness.sh](./eval_robustness.sh) — Jailbreak robustness evaluation with six attacks: **GCG**, **BEAST**, **GCQ**, **AutoDAN-Zhu**, **PAIR**, and **DeepInception**.
- [scripts/eval_utility.sh](./eval_utility.sh) — Utility evaluation by generating responses for [AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval).

All scripts support seven base models: `vicuna`, `mistral`, `llama2`, `qwen2.5`, `llama3.1`, and `gemma`. Below is a step-by-step tutorial demonstrating how to run experiments on the **Vicuna-7B-v1.5** model.


## Step 0: Enter the Scripts Folder

```bash
cd ./scripts
```


## Step 1: Adversarial Training

Train the model using either the **CAT** (Continuous Adversarial Training) or **ERCAT** (Embedding Regularized CAT, ours) method:

```bash
# Train with ERCAT (ours)
bash train.sh --model vicuna --method ercat

# Train with CAT (baseline)
bash train.sh --model vicuna --method cat
```

**Arguments:**

| Argument | Required | Choices | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | `vicuna`, `mistral`, `llama2`, `qwen2.5`, `llama3.1`, `gemma` | Base model to train |
| `--method` | Yes | `cat`, `ercat` | Training method |

Training results (LoRA adapter, trainer state, logs) will be saved to `./results/{model}/{method}/`.


## Step 2: Jailbreak Robustness Evaluation

Evaluate robustness by generating adversarial responses via jailbreak attacks and computing the Attack Success Rate (ASR) using the [HarmBench LLM judge](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls).

- **For a vanilla (pre-trained) base model** (no `--method`):

  ```bash
  bash eval_robustness.sh --model vicuna --attack gcg-1 --dataset harmbench
  ```

- **For an adversarially trained model** (with `--method`):

  ```bash
  bash eval_robustness.sh --model vicuna --attack gcg-1 --dataset harmbench --method ercat
  ```

**Arguments:**

| Argument | Required | Choices | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | `vicuna`, `mistral`, `llama2`, `qwen2.5`, `llama3.1`, `gemma` | Base model to evaluate |
| `--attack` | Yes | `gcg-1`, `beast-1`, `gcq-1`, `autodan-zhu-1`, `pair-4`, `deepinception-1` | Jailbreak attack |
| `--dataset` | Yes | `advbench`, `harmbench` | Evaluation dataset |
| `--method` | No | `cat`, `ercat` | If provided, evaluates the finetuned model; otherwise evaluates the vanilla base model |

The script internally runs two steps:
1. **Generate adversarial responses** — applies the specified attack and produces model responses.
2. **Judge responses** — computes the ASR via the HarmBench LLM judge.

Results will be saved to:
- Vanilla model: `./results/{model}/vanilla/eval-{attack}/`
- Finetuned model: `./results/{model}/{method}/eval-{attack}/`


## Step 3: Utility Evaluation (AlpacaEval)

Generate benign responses for [AlpacaEval2](https://github.com/tatsu-lab/alpaca_eval) utility evaluation:

- **For a vanilla (pre-trained) base model:**

  ```bash
  # Generated responses will be saved to:
  # ./results/vicuna/vanilla/eval-alpacaeval/alpacaeval_alpacaeval.json
  bash eval_utility.sh --model vicuna
  ```

- **For an adversarially trained model:**

  ```bash
  # Generated responses will be saved to:
  # ./results/vicuna/ercat/eval-alpacaeval/alpacaeval_alpacaeval.json
  bash eval_utility.sh --model vicuna --method ercat
  ```

**Arguments:**

| Argument | Required | Choices | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | `vicuna`, `mistral`, `llama2`, `qwen2.5`, `llama3.1`, `gemma` | Base model to evaluate |
| `--method` | No | `cat`, `ercat` | If provided, evaluates the finetuned model; otherwise evaluates the vanilla base model |

The utility evaluation is then performed based on the generated JSON file. See the [official tutorial](https://github.com/tatsu-lab/alpaca_eval?tab=readme-ov-file#evaluating-a-model) of AlpacaEval2 for detailed instructions.


## Supported Models

| Short Name | HuggingFace Model ID |
|------------|---------------------|
| `vicuna`   | `lmsys/vicuna-7b-v1.5` |
| `mistral`  | `mistralai/Mistral-7B-Instruct-v0.3` |
| `llama2`   | `meta-llama/Llama-2-7b-chat-hf` |
| `qwen2.5`  | `Qwen/Qwen2.5-7B-Instruct` |
| `llama3.1` | `meta-llama/Llama-3.1-8B-Instruct` |
| `gemma`    | `google/gemma-1.1-2b-it` |


## Results Directory Structure

```
results/
└── {model}/
    ├── vanilla/                        # Vanilla (pre-trained) model evaluations
    │   ├── eval-alpacaeval/            # Utility evaluation results
    │   ├── eval-gcg-1/                 # GCG attack results
    │   ├── eval-beast-1/               # BEAST attack results
    │   └── ...
    ├── cat/                            # CAT training & evaluation results
    │   ├── train_fin-model/            # Trained LoRA adapter
    │   ├── train_fin-trainer.pkl       # Trainer state
    │   ├── train_logs.jsonl            # Training logs
    │   ├── eval-alpacaeval/            # Utility evaluation results
    │   ├── eval-gcg-1/                 # GCG attack results
    │   └── ...
    └── ercat/                          # ERCAT training & evaluation results
        ├── train_fin-model/            # Trained LoRA adapter
        ├── ...
        └── eval-{attack}/             # Attack evaluation results
```
