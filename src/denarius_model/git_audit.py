"""Audit public candidates; the synthetic exception is exact and reproducible."""
import json
from pathlib import Path
from . import ROOT
from .synthetic import audit_public_dataset, public_payloads

def audit_candidates(files,root=ROOT):
    payloads,_=public_payloads()
    allowed={'data/synthetic/'+p for p in payloads}|{'data/synthetic/README.md'}
    bad=[]
    for name in files:
        path=Path(name)
        synthetic=name in allowed
        if path.parts[0] in ('models','outputs','reports','data') and name!='reports/evaluation.example.md' and path.name!='.gitkeep' and not synthetic:
            bad.append(name)
        if path.suffix in ('.csv','.tsv','.json','.jsonl','.parquet','.xlsx','.gguf','.bin','.safetensors','.db','.log','.key','.pem') and not synthetic:
            bad.append(name)
        if path.name.startswith('.env'): bad.append(name)
        if path.name=='.gitkeep' and (root/path).read_bytes(): bad.append(name)
        if path.suffix=='.ipynb':
            notebook=json.loads((root/path).read_text(encoding='utf-8'))
            if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook['cells']): bad.append(name)
    if bad: raise ValueError('Ficheiros privados ou notebook executado: '+', '.join(sorted(set(bad))))
    audit_public_dataset()
    return len(files)
