import json
from pathlib import Path
import yaml
from . import ROOT

def load_config():
    return {name: yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text(encoding='utf-8'))
            for name in ("model", "training", "dataset")}

def dataset_path(relative):
    return ROOT/'data/synthetic'/relative

def assert_dataset_version(experiment):
    current=read_json(dataset_path('processed/statistics.json'))['dataset_version']
    manifest=read_json(ROOT/'outputs'/experiment/'experiment.json')
    if manifest['dataset_version']!=current or manifest.get('dataset_source')!='synthetic':
        raise ValueError('Modelo treinado com outro dataset; escolher a experiência sintética correta')

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def model_source(config,tokenizer=False):
    if config.get('local_path'):
        path=Path(config['local_path']).expanduser()
        if not path.is_absolute(): path=ROOT/path
        if not path.is_dir(): raise FileNotFoundError(f'Pasta de modelo inexistente: {path}')
        return str(path)
    return config['tokenizer' if tokenizer else 'model_id']

def model_fingerprint(config):
    """Identifica pesos e tokenizer locais quando não existe commit HF resolvido."""
    if not config.get('local_path'): return None
    import hashlib
    path=Path(model_source(config))
    fingerprint={}
    for file in sorted(path.iterdir()):
        if file.is_file() and (file.suffix in ('.safetensors','.bin','.model') or file.name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','generation_config.json')):
            digest=hashlib.sha256()
            with file.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
            fingerprint[file.name]=digest.hexdigest()
    return fingerprint
