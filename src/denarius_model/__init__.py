"""Dados e experiências exclusivamente locais."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for key, value in {"HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true",
                   "HF_HOME": str(ROOT / ".cache/huggingface"),
                   "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[key] = value
