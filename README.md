# Understanding and Improving Continuous Adversarial Training for LLMs via In-Context Learning Theory

This is the official repository for the ICLR 2026 paper [**"Understanding and Improving Continuous Adversarial Training for LLMs via In-Context Learning Theory"**](https://openreview.net/forum?id=7zztxcmlyZ) by Shaopeng Fu and Di Wang.

## Overview

Large Language Models (LLMs) are vulnerable to jailbreak attacks that use adversarial prompts to induce undesirable or harmful outputs. Continuous Adversarial Training (CAT) is a computationally efficient defense that perturbs token embeddings in the continuous space rather than searching the discrete token space. This paper:

- Provides **theoretical analysis** connecting CAT's effectiveness to in-context learning (ICL) theory, showing that model robustness is linked to the singular values of the embedding matrix and the perturbation radius.
- Proposes **ERCAT** (Embedding Regularized CAT), which adds an embedding matrix regularization term to improve robustness beyond standard CAT.
- Validates the approach **empirically** across seven base models and six jailbreak attacks (GCG, BEAST, GCQ, AutoDAN-Zhu, PAIR, DeepInception).

## Installation

- Python 3.11
- CUDA 11.8
- PyTorch 2.5.1

### Build environment via Anaconda

Download and install [Anaconda3](https://www.anaconda.com/download). Then, run following commands:

```bash
# create & activate conda environment
conda create -n adv-ICL python=3.11
conda activate adv-ICL

# install packages
conda install pytorch=2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --upgrade peft==0.14.0 safetensors==0.4.5 datasets==3.2.0 accelerate==1.2.1 protobuf==5.29.1 sentencepiece==0.2.0 bitsandbytes==0.45.0

# for AlpacaEval evaluation
pip install alpaca-eval==0.6.6
```

### Build environment via Docker

The docker building file is [./Dockerfile](./Dockerfile). Run following commands, and then the built image is continuous-adv-icl:latest.

```bash
docker pull pytorch/pytorch:2.5.1-cuda11.8-cudnn9-devel
docker build --tag 'continuous-adv-icl' .
```

**PS:** If you plan to use Docker to run your experiments, don't forget to **mount your default cache folder (e.g., `${HOME}/.cache`) to `/root/.cache` in the Docker container**.

## Quick Start

- [./src](./src) stores all experiment source codes.
- [./configs](./configs) stores all configuration files for experiments.
- [./scripts](./scripts) stores all scripts for running experiments.
- [./data](./data) stores the evaluation datasets.

Please see the [scripts/README.md](./scripts/README.md) for the tutorial on how to run experiments.

## Citation

```
@inproceedings{fu2026understanding,
  title={Understanding and Improving Continuous Adversarial Training for {LLM}s via In-Context Learning Theory},
  author={Shaopeng Fu and Di Wang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=7zztxcmlyZ}
}
```

## Acknowledgment

- AdvBench dataset: [https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)
- HarmBench dataset: [https://github.com/centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)
