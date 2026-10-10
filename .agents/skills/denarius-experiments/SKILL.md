---
name: denarius-experiments
description: Retomar treino LoRA ou QLoRA e avaliação de classificação Denarius no corpus sintético versionado, comparando baselines e analisando erros.
---

# Experiências sintéticas locais

Ler configs/model.yaml, training.yaml e dataset.yaml, módulos train/evaluate/
inference e skill denarius-dataset se alterar dados. O dataset ativo é só
 data/synthetic; não ler movimentos reais ou chamar containers.

## Modelos e ambiente

Model ID Qwen/Qwen3-0.6B-Base; local_path pode referir pesos HF fora do repo.
Verificar config/tokenizer/safetensors. GGUF qwen3:0.6b de Ollama é outro baseline,
com template nativo e thinking=false; não substitui pesos de treino PEFT.
Manter local_files_only=true. Diagnosticar scripts/diagnose.py. Alvo RTX 4050 6 GB,
CUDA PyTorch e BF16. Python 3.10 não é suportado; ambiente inicial verificado 3.14.
Kernel deve usar o mesmo Python onde pip instalou o projeto.

LoRA inicial rank 8, batch 1, acumulação 16, checkpointing, três épocas e early
stopping por validation loss completo. QLoRA requer bitsandbytes CUDA testado.
SFT usa Transformers Trainer + PEFT, prompt labels -100, loss na completion,
report_to=none, push_to_hub=False. Não depende de TRL. Smoke sintético minúsculo
scripts/smoke_lora.py verifica APIs, não memória/qualidade do treino 0.6B.
Verificar tokens; não truncar lista de categorias ou resposta para caber no contexto.

## Continuidade e avaliação

IDs novos synthetic-001, synthetic-002, etc. Os modelos e métricas antigos de
candidate-001 eram de dados reais: não usar como resultados deste corpus.
Manifest local regista dataset_source=synthetic, version, seed, configs, hashes
locais de base/tokenizer, revision resolvida, versões e métricas. Recusar adapters
ou relatórios com outro dataset/pesos. Não sobrescrever uma experiência válida.

Notebook prepara corpus e orquestra baselines/treino/validação. Depois de alterar
código ou notebook, reiniciar kernel e reabrir para evitar funções antigas.
Ollama baseline é opcional, por omissão False. RUN_FULL_VALIDATION=False usa
amostra determinística de dois exemplos por categoria apenas para desenvolvimento;
comparar mesma sample_size/evaluation_rows_hash. CLI sem --per-category usa
validação completa. Testes finais sempre completos, sem amostragem.

Baselines: scripts/evaluate.py --backend ml, base ou ollama-base. TF-IDF fit só
no treino. Treino: scripts/train.py --experiment synthetic-001. Tuned: --backend
tuned. scripts/robustness.py mede ablações; montante/direção podem justificar
outra categoria, portanto mudança de predição não implica erro automaticamente.

Congelar candidato antes de --split test --final ou --split hard_test --final.
Não usar os testes para tuning. Ambos são sintéticos: accuracy alta não prova
qualidade em extratos reais, nem autoriza integração. Futuro teste real privado
precisa de pedido explícito; não recuperar dados reais automaticamente.

Output só JSON category válido, sem reasoning/confidence/Markdown ou texto extra.
Inválidos contam como erros. Consultar macro/weighted F1, precision/recall por
classe, suporte, matriz e erros. Report scripts/report.py exclui outros datasets;
analyze_errors.py permite revisão local. Modelos, outputs e relatórios seguem
ignorados, mesmo quando usam dados sintéticos. Guardar notebook sem outputs.
