import time
import json
import urllib.request
import urllib.parse
from .config import model_source,model_fingerprint,assert_dataset_version
from .prompting import prompt, schema, parse_output

class HFClassifier:
    def __init__(self, config, adapter=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        if adapter:
            from .config import read_json
            from . import ROOT
            from pathlib import Path
            manifest=read_json(ROOT/'outputs'/Path(adapter).parent.name/'experiment.json')
            assert_dataset_version(Path(adapter).parent.name)
            config={**manifest['configuration']['model'], 'revision':manifest['resolved_revision'] or manifest['configuration']['model']['revision']}
        self.cfg=config
        fingerprint=model_fingerprint(config)
        if adapter and manifest.get('model_fingerprint')!=fingerprint:
            raise ValueError('Pesos locais alterados desde o treino')
        self.tokenizer=AutoTokenizer.from_pretrained(model_source(config,True),revision=config['revision'],local_files_only=config['local_files_only'])
        self.model=AutoModelForCausalLM.from_pretrained(model_source(config),revision=config['revision'],local_files_only=config['local_files_only'],dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,device_map='auto')
        if adapter: self.model=PeftModel.from_pretrained(self.model,adapter)
        self.model.eval()
        self.metadata={'resolved_revision':getattr(self.model.config,'_commit_hash',None),'configuration':config,'model_fingerprint':fingerprint,'decoding':'greedy unconstrained'}
    def __call__(self,row,categories):
        import torch
        inputs=self.tokenizer(prompt(row,categories),add_special_tokens=False,return_tensors='pt').to(self.model.device)
        if inputs.input_ids.shape[1]+self.cfg['max_new_tokens']>self.cfg['max_length']:
            raise ValueError('Prompt excede o contexto configurado; não truncar categorias')
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            ids=self.model.generate(**inputs,max_new_tokens=self.cfg['max_new_tokens'],do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        text=self.tokenizer.decode(ids[0,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        return text,time.perf_counter()-start,{'tokens':ids.shape[1]-inputs.input_ids.shape[1],
            'peak_vram_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}

class OllamaClassifier:
    def __init__(self,model='denarius-movement',url='http://127.0.0.1:11434',native_template=False):
        if urllib.parse.urlparse(url).hostname not in ('localhost','127.0.0.1','::1'):
            raise ValueError('Apenas Ollama local é permitido')
        self.model=model; self.url=url; self.native_template=native_template
        self.metadata={'ollama_model':model,'decoding':'schema constrained','native_template':native_template,'thinking':False}
    def unload(self):
        request=urllib.request.Request(self.url+'/api/generate',data=json.dumps({'model':self.model,'keep_alive':0}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response: json.load(response)
    def __call__(self,row,categories):
        body={'model':self.model,'prompt':prompt(row,categories),'raw':not self.native_template,'stream':False,'think':False,
              'format':schema(categories),'options':{'temperature':0,'seed':42,'num_predict':128,'num_ctx':2048}}
        start=time.perf_counter()
        request=urllib.request.Request(self.url+'/api/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=120) as response: value=json.load(response)
        return value['response'],time.perf_counter()-start,{'tokens':value.get('eval_count'),
            'tokens_per_second':value.get('eval_count',0)/(value.get('eval_duration',1)/1e9)}

def classify(classifier,row,categories):
    return {'category':parse_output(classifier(row,categories)[0],categories)}
