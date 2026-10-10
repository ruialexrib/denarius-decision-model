"""Benchmark quente local, validação por omissão, mesmo schema em F16 e Q4."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import subprocess
from denarius_model import ROOT
from denarius_model.config import read_json,save_json,dataset_path
from denarius_model.inference import OllamaClassifier
from denarius_model.evaluate import evaluate

p=argparse.ArgumentParser()
p.add_argument('--model',default='denarius-movement')
p.add_argument('--experiment',default='synthetic-001')
p.add_argument('--repeats',type=int,default=3)
a=p.parse_args()
classifier=OllamaClassifier(a.model)
rows=read_json(dataset_path('splits/validation.json'))
cats=read_json(dataset_path('processed/categories.json'))
_,cold,_=classifier(rows[0],cats)
results=[]
for i in range(a.repeats): results.append(evaluate(classifier,'validation',f'ollama-{a.model}-warm-{i}',a.experiment))
try:
    vram=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
except (FileNotFoundError,subprocess.CalledProcessError): vram=None
save_json(ROOT/'reports'/a.experiment/f'{a.model}_benchmark.json',{'cold_first_request_seconds':cold,'warm_runs':results,'gpu_total_used_mib_after_run':vram,'note':'VRAM total da GPU, não isolada do processo Ollama; inclui outras aplicações.'})
print('Benchmark guardado localmente.')
