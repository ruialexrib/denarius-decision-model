import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from denarius_model import ROOT
from denarius_model.config import read_json
from collections import Counter
import argparse
p=argparse.ArgumentParser()
p.add_argument('--experiment',default='synthetic-001')
p.add_argument('--backend',default='tuned')
a=p.parse_args()
folder=ROOT/'reports'/a.experiment
rows=read_json(folder/f'{a.backend}_validation_predictions.json')
confusions=Counter((r['row']['category'],r['prediction']) for r in rows if r['row']['category']!=r['prediction'])
lines=['# Erros de validação','', 'Corpus sintético; consultar previsões locais para exemplos de erros.', '']
for (truth,pred),count in confusions.most_common(): lines.append(f'- {truth} → {pred}: {count}')
(folder/f'{a.backend}_errors.md').write_text('\n'.join(lines),encoding='utf-8')
