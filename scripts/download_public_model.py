"""Apenas download de pesos públicos: não lê datasets ou outputs de treino."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from denarius_model.config import load_config,save_json
from denarius_model import ROOT
from huggingface_hub import snapshot_download
cfg=load_config()['model']
paths={}
for repo in set([cfg['model_id'],cfg['tokenizer']]):
    paths[repo]=snapshot_download(repo_id=repo,revision=cfg['revision'],allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'],local_files_only=False)
save_json(ROOT/'outputs/public_model_download.json',paths)
print('Pesos públicos guardados no cache privado. Fixar revision ao SHA da pasta snapshots para reprodutibilidade.')
