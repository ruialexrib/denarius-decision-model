import argparse
import json
from .config import load_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['diagnose','generate','prepare','train','evaluate','export','report'])
    p.add_argument('--experiment',default='synthetic-001')
    p.add_argument('--backend',choices=['base','tuned','ml','ollama','ollama-base'],default='base')
    p.add_argument('--split',choices=['validation','test','hard_test'],default='validation')
    p.add_argument('--final',action='store_true')
    p.add_argument('--per-category',type=int,default=None,help='Amostra por categoria, apenas validação de desenvolvimento')
    a=p.parse_args(); cfg=load_config()
    if a.command=='diagnose':
        from .diagnostics import diagnose
        print(json.dumps(diagnose(),indent=2))
    elif a.command=='generate':
        from .synthetic import generate_dataset
        print('Dataset sintético:',generate_dataset()['sizes'])
    elif a.command=='prepare':
        from .dataset import prepare
        result=prepare(); print('Splits:',result['sizes'])
    elif a.command=='train':
        from .train import train
        train(cfg,a.experiment)
    elif a.command=='export':
        from .export import merge
        print(merge(cfg,a.experiment))
    elif a.command=='report':
        from .evaluate import report
        report(a.experiment)
    else:
        from . import ROOT
        from .evaluate import evaluate,ml_baseline
        from .inference import HFClassifier,OllamaClassifier
        predictor=ml_baseline() if a.backend=='ml' else OllamaClassifier() if a.backend=='ollama' else OllamaClassifier(cfg['model']['ollama_baseline'],native_template=True) if a.backend=='ollama-base' else HFClassifier(cfg['model'],str(ROOT/'models'/a.experiment/'adapter') if a.backend=='tuned' else None)
        m=evaluate(predictor,a.split,a.backend,a.experiment,a.final,per_category=a.per_category)
        if isinstance(predictor,OllamaClassifier): predictor.unload()
        print({k:m[k] for k in ('accuracy','macro_f1','weighted_f1','invalid_outputs')})

if __name__=='__main__': main()
