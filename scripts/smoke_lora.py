"""Valida Trainer + PEFT + CUDA num Qwen3 minúsculo aleatório, sem downloads/dados."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from denarius_model import ROOT
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,Trainer,TrainingArguments,default_data_collator
from peft import LoraConfig,get_peft_model
from datasets import Dataset

if not torch.cuda.is_available(): raise SystemExit('CUDA indisponível')
config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=16)
model=Qwen3ForCausalLM(config)
model.config.use_cache=False
model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
sample={'input_ids':[1,2,3,4,5,6,7,8],'attention_mask':[1]*8,'labels':[-100]*4+[5,6,7,8]}
dataset=Dataset.from_list([sample,sample])
args=TrainingArguments(output_dir=str(ROOT/'outputs/synthetic-smoke'),max_steps=1,per_device_train_batch_size=1,bf16=torch.cuda.is_bf16_supported(),gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},save_strategy='no',report_to='none',push_to_hub=False,disable_tqdm=True)
trainer=Trainer(model=model,args=args,train_dataset=dataset,data_collator=default_data_collator)
trainer.train()
assert any(p.grad is not None for p in model.parameters()) or trainer.state.global_step==1
print('Smoke CUDA/PEFT/Trainer concluído. Não avalia qualidade nem VRAM do modelo 0.6B.')
