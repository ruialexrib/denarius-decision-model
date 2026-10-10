"""Sensibilidade na validação; não usa teste final para desenvolvimento."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from denarius_model import ROOT
from denarius_model.config import load_config,read_json,save_json,dataset_path
from denarius_model.inference import HFClassifier,OllamaClassifier
from denarius_model.prompting import parse_output

p=argparse.ArgumentParser()
p.add_argument('--experiment',default='synthetic-001')
p.add_argument('--backend',choices=['base','tuned','ollama'],default='tuned')
a=p.parse_args(); cfg=load_config()
classifier=OllamaClassifier() if a.backend=='ollama' else HFClassifier(cfg['model'],str(ROOT/'models'/a.experiment/'adapter') if a.backend=='tuned' else None)
rows=read_json(dataset_path('splits/validation.json')); cats=read_json(dataset_path('processed/categories.json'))
def predict(row):
    output=classifier(row,cats)[0]
    try: return parse_output(output,cats)
    except (ValueError,TypeError): return '__INVALID__'
details=[]
for row in rows:
    original=predict(row)
    variants={'amount_zero':{**row,'amount':'0.00'},
              'direction_flip':{**row,'direction':'credito_contabilistico' if row['direction']=='debito_contabilistico' else 'debito_contabilistico'},
              'text_case':{**row,'description':row['description'].lower()},
              'text_spaces':{**row,'description':'  '.join(row['description'].split())}}
    details.append({'original':original,'truth':row['category'],'variants':{k:predict(v) for k,v in variants.items()}})
summary={k:{'prediction_changed_fraction':sum(r['variants'][k]!=r['original'] for r in details)/len(details),'accuracy_against_original_label':sum(r['variants'][k]==r['truth'] for r in details)/len(details)} for k in details[0]['variants']}
save_json(ROOT/'reports'/a.experiment/f'{a.backend}_robustness.json',{'summary':summary,'details':details,'note':'Ablações são contrafactuais: mudança de direção/montante pode justificar outra categoria; taxa de mudança não equivale a erro.'})
print('Robustez guardada localmente.')
