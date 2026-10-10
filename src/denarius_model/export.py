from . import ROOT
from .config import read_json,model_source,model_fingerprint,assert_dataset_version

def merge(config,experiment):
    assert_dataset_version(experiment)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    manifest=read_json(ROOT/'outputs'/experiment/'experiment.json')
    cfg=manifest['configuration']['model']; revision=manifest['resolved_revision'] or cfg['revision']
    if manifest.get('model_fingerprint')!=model_fingerprint(cfg): raise ValueError('Base local alterada desde treino')
    model=AutoModelForCausalLM.from_pretrained(model_source(cfg),revision=revision,local_files_only=True,dtype=torch.float32,device_map='cpu')
    model=PeftModel.from_pretrained(model,ROOT/'models'/experiment/'adapter').merge_and_unload()
    out=ROOT/'models'/experiment/'merged'; model.save_pretrained(out,safe_serialization=True)
    AutoTokenizer.from_pretrained(ROOT/'models'/experiment/'adapter',local_files_only=True).save_pretrained(out)
    return out
