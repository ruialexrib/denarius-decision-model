import json
from collections import Counter
from . import ROOT
from .config import read_json,save_json,dataset_path,assert_dataset_version
from .prompting import parse_output

def evaluate(classifier, split, name, experiment, final=False, per_category=None):
    import numpy as np
    from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
    if split not in ('validation','test','hard_test'): raise ValueError('Split inválido')
    if split!='validation' and not final: raise ValueError('Teste final exige --final')
    if per_category is not None and (split!='validation' or type(per_category) is not int or per_category<1):
        raise ValueError('Amostragem positiva só é permitida na validação de desenvolvimento')
    rows=read_json(dataset_path(f'splits/{split}.json')); categories=read_json(dataset_path('processed/categories.json'))
    import random,hashlib
    if per_category is not None:
        rng=random.Random(42)
        rows=[r for category in categories for r in rng.sample([r for r in rows if r['category']==category],min(per_category,sum(r['category']==category for r in rows)))]
    if name=='tuned':
        assert_dataset_version(experiment)
    predictions=[]; times=[]; details=[]
    for row in rows:
        output,elapsed,perf=classifier(row,categories)
        try: predicted=parse_output(output,categories)
        except (ValueError,TypeError): predicted='__INVALID__'
        predictions.append(predicted); times.append(elapsed)
        details.append({'row':row,'prediction':predicted,'raw_output':output,'seconds':elapsed,**perf})
    truth=[r['category'] for r in rows]
    train_counts=Counter(r['category'] for r in read_json(dataset_path('splits/train.json')))
    metrics={'accuracy':accuracy_score(truth,predictions),'macro_f1':f1_score(truth,predictions,labels=categories,average='macro',zero_division=0),'weighted_f1':f1_score(truth,predictions,labels=categories,average='weighted',zero_division=0),'invalid_outputs':predictions.count('__INVALID__'),'per_category':classification_report(truth,predictions,labels=categories,output_dict=True,zero_division=0),'confusion_labels':categories+['__INVALID__'],'confusion_matrix':confusion_matrix(truth,predictions,labels=categories+['__INVALID__']).tolist(),'latency':{'mean':float(np.mean(times)),'median':float(np.median(times)),'p95':float(np.percentile(times,95))},'unseen_training_labels':sorted(set(truth)-set(train_counts)),'metadata':getattr(classifier,'metadata',{}),'dataset_version':read_json(dataset_path('processed/statistics.json'))['dataset_version'],'dataset_source':'synthetic','sample_size':len(rows)}
    metrics['seed']=42
    metrics['evaluation_rows_hash']=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    metrics['development_per_category']=per_category
    metrics['sizes']=read_json(dataset_path('processed/statistics.json'))['sizes']
    metrics['error_counts']=dict(Counter(r['category'] for r,p in zip(rows,predictions) if r['category']!=p))
    import datetime
    metrics['timestamp']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path=ROOT/'reports'/experiment
    save_json(path/f'{name}_{split}.json',metrics); save_json(path/f'{name}_{split}_predictions.json',details)
    return metrics

def ml_baseline():
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import time
    rows=read_json(dataset_path('splits/train.json'))
    pipeline=make_pipeline(TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))
    pipeline.fit([r['description'] for r in rows],[r['category'] for r in rows])
    def predict(row,categories):
        start=time.perf_counter(); category=pipeline.predict([row['description']])[0]
        return json.dumps({'category':category}),time.perf_counter()-start,{}
    return predict

def report(experiment):
    path=ROOT/'reports'/experiment
    path.mkdir(parents=True,exist_ok=True)
    version=read_json(dataset_path('processed/statistics.json'))['dataset_version']
    lines=['# Avaliação sintética local','', 'Resultados deste corpus artificial não demonstram desempenho em movimentos reais.',f'Dataset: {version}', '', '| Modelo/split | N | Accuracy | Macro F1 | Weighted F1 | Inválidos | p95 s |','|---|---:|---:|---:|---:|---:|---:|']
    for file in sorted(path.glob('*.json')):
        if file.stem.endswith(('predictions','benchmark','robustness')): continue
        m=read_json(file)
        if m.get('dataset_version')!=version or m.get('dataset_source')!='synthetic': continue
        lines.append(f"| {file.stem} | {m['sample_size']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['weighted_f1']:.3f} | {m['invalid_outputs']} | {m['latency']['p95']:.3f} |")
    lines+=['','Comparar modelos com o mesmo evaluation_rows_hash e sample_size; amostras de desenvolvimento não são avaliação final.']
    lines+=['','Consultar JSON para suporte, categorias ausentes de treino e matriz de confusão.','HF usa geração livre; Ollama usa schema: separar efeito da quantização do efeito de decoding.','Latências incluem primeiro pedido frio; repetir após aquecimento para medir regime estável.','Não declarar prontidão sem rever erros, categorias raras e generalização.']
    (path/'evaluation.md').write_text('\n'.join(lines),encoding='utf-8')
