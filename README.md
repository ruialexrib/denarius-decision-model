<div align="center">

# Denarius Decision Model

### Financial classification, researched locally.

[![Python](https://img.shields.io/badge/Python-3.11–3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qwen3](https://img.shields.io/badge/Qwen3-0.6B_Base-6C63FF)](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA_%2F_QLoRA-FFD21E)](https://huggingface.co/docs/peft/index)
[![Ollama](https://img.shields.io/badge/Ollama-Local-159A70)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Developed by [Rui Ribeiro](https://github.com/ruialexrib)

</div>

---

## About

An independent research project for local financial classification with **Qwen3-0.6B-Base**, **LoRA/QLoRA** and export to **Ollama**. It uses **12,600 synthetic examples** covering the **70 existing DenariusAI categories**.

The notebook covers data exploration, baseline comparison, fine-tuning and evaluation. Descriptions and category labels remain in Portuguese. Results on synthetic data do not demonstrate performance on real transactions.

## Quick start

Use Python **3.11–3.14** and a CUDA-enabled PyTorch installation for GPU training. The target GPU is an RTX 4050 with 6 GB VRAM.

From the project root:

```powershell
python -m pip install -e ".[test]"
python -m jupyter lab
```

Set `local_path` in [configs/model.yaml](configs/model.yaml) to your Hugging Face model directory. The default is `~/models/Qwen3-0.6B-Base`.

Open [the notebook](notebooks/01_train_and_evaluate.ipynb), select the Python environment where dependencies were installed, and run the cells in order. Restart the kernel after installing dependencies. The notebook provides the training workflow; running the scripts separately is optional.

Final testing and export are disabled by default. Enable them after reviewing validation results. GGUF conversion and Ollama model creation are separate steps.

The notebook starts with `QUICK_RUN=True`: 140 training examples, one epoch and a small validation sample for a faster demonstration. To train with all data, set `QUICK_RUN=False`, restart the kernel and run the cells again. Quick and full runs use separate experiment IDs.

## Dataset

The versioned corpus contains 8,400 training examples and 1,400 examples in each validation, test and hard-test split. It is entirely synthetic and requires no database or container access.

See the [dataset card](data/synthetic/README.md) for provenance, split rules and limitations. To regenerate:

```powershell
python scripts/generate_dataset.py
```

Real financial data, model weights, adapters and generated results are excluded from Git. Commit notebooks without saved outputs.

## Project structure

| Folder | Purpose |
|---|---|
| `notebooks/` | Training and evaluation workflow |
| `src/` and `scripts/` | Pipeline and command-line tools |
| `configs/` | Model, training and dataset settings |
| `data/synthetic/` | Public synthetic corpus |
| `tests/` | Pipeline and privacy checks |
| `.agents/skills/` | Guidance for continuing the project |

Before committing, run `python scripts/audit_git.py` and `python -m pytest -q`.

## License

Distributed under the [MIT License](LICENSE).
