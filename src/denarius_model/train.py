from . import ROOT
from .config import read_json,save_json,model_source,model_fingerprint,dataset_path
from .prompting import example

def training_rows(config, split):
    """Select a deterministic, class-balanced development subset without changing files."""
    if split not in ('train', 'validation'):
        raise ValueError('Training subsets only support train and validation')
    rows = read_json(dataset_path(f'splits/{split}.json'))
    count = config['training'].get(f'{split}_per_category')
    if count is None:
        return rows
    if type(count) is not int or count < 1:
        raise ValueError('Examples per category must be a positive integer')
    import random
    rng = random.Random(config['training']['seed'])
    selected = []
    for category in sorted({r['category'] for r in rows}):
        candidates = [r for r in rows if r['category'] == category]
        if count > len(candidates):
            raise ValueError('Requested subset exceeds available examples')
        selected.extend(rng.sample(candidates, count))
    rng.shuffle(selected)
    return selected

def train(config,experiment):
    if (ROOT/'outputs'/experiment/'experiment.json').exists() or (ROOT/'models'/experiment/'adapter').exists():
        raise ValueError('Experiência existente; escolher um identificador novo')
    from .synthetic import audit_public_dataset
    audit_public_dataset()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForSeq2Seq, BitsAndBytesConfig, set_seed
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    cfg=config['training']; modelcfg=config['model']; set_seed(cfg['seed'])
    fingerprint=model_fingerprint(modelcfg)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA obrigatório para este perfil')
    tokenizer=AutoTokenizer.from_pretrained(model_source(modelcfg,True),revision=modelcfg['revision'],local_files_only=modelcfg['local_files_only'])
    tokenizer.pad_token=tokenizer.eos_token
    categories=read_json(dataset_path('processed/categories.json'))
    def encode(row):
        ex=example(row,categories)
        prefix=tokenizer.encode(ex['prompt'],add_special_tokens=False)
        completion=tokenizer.encode(ex['completion'],add_special_tokens=False)+[tokenizer.eos_token_id]
        if len(prefix)+len(completion)>modelcfg['max_length']: raise ValueError('Exemplo excede max_length')
        ids=prefix+completion
        return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prefix)+completion}
    selected_rows={s:training_rows(config,s) for s in ('train','validation')}
    sets={s:Dataset.from_list([encode(r) for r in selected_rows[s]]) for s in selected_rows}
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs=dict(revision=modelcfg['revision'],local_files_only=modelcfg['local_files_only'],dtype=dtype)
    if cfg['strategy']=='qlora':
        kwargs.update(device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype))
    elif cfg['strategy']!='lora': raise ValueError('Estratégia inválida')
    model=AutoModelForCausalLM.from_pretrained(model_source(modelcfg),**kwargs)
    revision=getattr(model.config,'_commit_hash',None)
    if cfg['strategy']=='qlora': model=prepare_model_for_kbit_training(model)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=cfg['dropout'],target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    out=ROOT/'outputs'/experiment
    args=TrainingArguments(output_dir=str(out),num_train_epochs=cfg['epochs'],learning_rate=cfg['learning_rate'],per_device_train_batch_size=cfg['batch_size'],per_device_eval_batch_size=1,gradient_accumulation_steps=cfg['gradient_accumulation'],gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},bf16=dtype==torch.bfloat16,fp16=dtype==torch.float16,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',greater_is_better=False,report_to='none',push_to_hub=False,seed=cfg['seed'],logging_steps=5)
    trainer=Trainer(model=model,args=args,train_dataset=sets['train'],eval_dataset=sets['validation'],data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True),callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    trainer.train()
    adapter=ROOT/'models'/experiment/'adapter'; model.save_pretrained(adapter); tokenizer.save_pretrained(adapter)
    stats=read_json(dataset_path('processed/statistics.json'))
    import datetime,importlib.metadata
    manifest={'experiment_id':experiment,'configuration':config,'resolved_revision':revision,'dataset_version':stats['dataset_version'],'sizes':stats['sizes'],'seed':cfg['seed'],'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metrics':trainer.state.log_history,'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate','datasets')}}
    manifest['model_fingerprint']=fingerprint
    manifest['dataset_source']='synthetic'
    import hashlib,json
    manifest['training_sizes']={s:len(rows) for s,rows in selected_rows.items()}
    manifest['training_rows_hashes']={s:hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest() for s,rows in selected_rows.items()}
    manifest['development_run']=any(cfg.get(f'{s}_per_category') is not None for s in selected_rows)
    save_json(out/'experiment.json',manifest)
    return str(adapter)
