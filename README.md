<div align="center">

# Denarius Decision Model

### Classificação financeira, investigada localmente.

[![Python](https://img.shields.io/badge/Python-3.11–3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen3](https://img.shields.io/badge/Qwen3-0.6B_Base-6C63FF)](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA_%2F_QLoRA-FFD21E)](https://huggingface.co/docs/peft/index)
[![Ollama](https://img.shields.io/badge/Ollama-Local-159A70)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Desenvolvido por [Rui Ribeiro](https://github.com/ruialexrib)

[Abrir o notebook](notebooks/01_train_and_evaluate.ipynb) · [Consultar o dataset](data/synthetic/README.md)

</div>

---

## Sobre

Projeto experimental independente para classificação financeira local com
Qwen3-0.6B-Base, LoRA/QLoRA e exportação para Ollama. O dataset atual é
**inteiramente sintético**, com as **70 categorias ativas existentes** do
DenariusAI, preservando o formato `grupo / categoria`.

Uma análise estrutural local única dos movimentos anteriores orientou os
formatos de texto. Todos os novos textos e montantes foram inventados. O
pipeline atual não consulta containers ou SQL nem lê os datasets privados.
Não existe integração na aplicação DenariusAI.

## Destaques

- **70 categorias DenariusAI** preservadas numa taxonomia versionada.
- **12 600 exemplos sintéticos**, equilibrados e separados por cenário.
- **Notebook completo** para explorar dados, comparar baselines, treinar e analisar erros.
- **Treino local LoRA/QLoRA**, com configurações e manifestos de experiências.
- **Avaliação estruturada**, incluindo validade JSON, métricas por categoria e latência.
- **Exportação para Ollama** através de merge, GGUF e quantização Q4_K_M.
- **Skills de continuidade** para retomar dados, experiências e exportação.

## Tecnologia

| Tecnologia | Função |
|---|---|
| **Python / Jupyter** | Pipeline e notebook de investigação |
| **Qwen3-0.6B-Base** | Modelo base para ajuste supervisionado |
| **PyTorch / Transformers** | Execução em GPU e treino |
| **PEFT / bitsandbytes** | Adapters LoRA e quantização opcional QLoRA |
| **scikit-learn** | Baseline TF-IDF, classificação e métricas |
| **llama.cpp / Ollama** | Conversão GGUF e inferência local |

## Início rápido

### Começar pelo notebook

Usar Python 3.11–3.14; nesta máquina foi verificado Python 3.14 com CUDA/PEFT.
Instalar no mesmo kernel que executará o notebook. Se a pasta atual é notebooks:

```python
%pip install -e "..[test]"
```

Ou, na raiz do projeto, em PowerShell:

```powershell
python -m pip install -e ".[test]"
python -m jupyter lab
```

Abrir `notebooks/01_train_and_evaluate.ipynb`, reiniciar o kernel se estava a
usar a versão anterior e executar as células por ordem. O notebook prepara
os dados sintéticos, explora exemplos, compara baselines, faz treino LoRA,
avalia validação, mostra métricas/curvas/erros e prepara a exportação.
**Não é preciso executar também os scripts para repetir o treino.**

O diagnóstico deve mostrar `cuda_available: true`. O kernel deve ter PyTorch
CUDA funcional; `pip install` não garante isso. Pesos já instalados nesta
máquina estão configurados por `local_path` em configs/model.yaml. Noutra
máquina, ajustar esse caminho ou descarregar pesos públicos para o cache.
Nenhum peso/modelo é incluído no Git.

Configuração inicial do notebook:
- `experiment = "synthetic-001"` separa esta experiência das experiências reais antigas.
- `RUN_OLLAMA_BASELINE=False`: baseline opcional, exige Ollama local em execução.
- `RUN_FULL_VALIDATION=False`: avaliação exploratória com dois exemplos por
  categoria; passar a True para avaliar os 1 400 exemplos de validação.
- `RUN_FINAL=False` e `RUN_EXPORT=False`: fases posteriores, após rever validação.

A amostra exploratória é a mesma para os modelos comparados. Resultados com
amostras diferentes não devem ser comparados diretamente; ficam registados
sample_size e evaluation_rows_hash. Mesmo na primeira execução, early stopping
usa validation loss no split de validação completo.

## Dataset e reprodução

O corpus já está em `data/synthetic/` e pode ser incluído no repositório:

| Split | Exemplos | Exemplos por categoria |
|---|---:|---:|
| Treino | 8 400 | 120 |
| Validação | 1 400 | 20 |
| Teste normal | 1 400 | 20 |
| Teste difícil | 1 400 | 20 |

`configs/taxonomy.yaml` congela as categorias ativas; categorias inativas não
entram no contrato. `configs/dataset.yaml` define seed e contagens. O vocabulário,
os intervalos inventados e as regras estão em src/denarius_model/synthetic.py.
Não precisa de LLM/cloud para gerar. Para regenerar:

```powershell
python scripts/generate_dataset.py
# Alternativa equivalente, que também verifica o corpus:
python scripts/prepare_dataset.py
```

A geração é determinística, grava manifest/hashes e valida cobertura de classes,
montantes, duplicados e separação de cenários. Mais exemplos exigem também
mais contextos/templates: o gerador rejeita quantidades superiores à diversidade
suportada. Não aumentar volume repetindo a mesma descrição com outros valores.

Raw sintético: data/synthetic/raw/movements.jsonl. Dados prontos:
data/synthetic/splits/{train,validation,test,hard_test}.json. Taxonomia/estatísticas:
data/synthetic/processed/. Proveniência e hashes: data/synthetic/manifest.json.
Ler [a ficha do dataset](data/synthetic/README.md) para convenções e limitações.

Um cenário é contraparte + contexto de uma categoria. Suas variantes ficam
no mesmo split e não há overlap de descrições canónicas. O teste normal pode
partilhar contrapartes com treino; o difícil tem contrapartes fictícias,
paráfrases e famílias de template novas. Marcas que fazem parte do label
(ALDI, Netflix, etc.) continuam conhecidas: o teste não é de cadeia comercial
inédita para essas categorias. Débito/crédito é contabilístico, não sinal bancário.

## Privacidade e Git

**Dados financeiros reais nunca devem ser committed. Modelos, adapters,
checkpoints e GGUF nunca devem ser committed.** O único dataset permitido
no Git é o corpus sintético exatamente reproduzível pelo gerador.
As exceções do .gitignore enumeram oito JSON/JSONL concretos e a ficha do corpus;
qualquer outro ficheiro em data continua ignorado. Os dados reais antigos em
data/raw, data/processed e data/splits permanecem locais e fora do pipeline.
Não foram apagados, não são reutilizados e não devem ser publicados.

Sem trackers remotos: telemetry HF desativada, report_to=none, push_to_hub=False.
Inferência Ollama só em loopback. Outputs/relatórios/modelos são ignorados;
notebooks no Git sem resultados salvos. Não há .env.example nem leitores Docker:
nenhuma credencial SQL é necessária para esta versão.

Antes de um commit:

```powershell
python scripts/audit_git.py
python -m pytest -q
git status --short
git diff --cached
```

A auditoria compara byte a byte o corpus com o gerador e rejeita paths privados,
ficheiros sintéticos alterados manualmente e notebooks executados. Não prova
que código/docs não contêm informação privada: rever também o conteúdo.
Não usar git add -f para contornar estas regras.

## Modelo, hardware e treino

Alvo RTX 4050 6 GB. Qwen/Qwen3-0.6B-Base configurável em configs/model.yaml;
local_files_only=true. local_path pode ser absoluto ou relativo à raiz e deve
conter config, tokenizer e safetensors. Para obter pesos públicos opcionalmente:
`python scripts/download_public_model.py`. Não abre dados de treino.

LoRA inicial: rank 8, batch 1, acumulação 16, gradient checkpointing, BF16 se
suportado, três épocas e early stopping por validation loss. Trainer + PEFT
implementa SFT com labels do prompt mascaradas (-100), loss na completion.
Não depende de TRL. Perfil QLoRA é opcional: instalar `.[qlora]` e testar CUDA
bitsandbytes antes de mudar strategy. Diagnóstico: scripts/diagnose.py; smoke
sintético de API/CUDA: scripts/smoke_lora.py. Smoke não mede memória do treino
0.6B. Não confundir CUDA do driver com runtime PyTorch.

```powershell
python scripts/train.py --experiment synthetic-001
python scripts/evaluate.py --backend tuned --experiment synthetic-001 --per-category 2
python scripts/evaluate.py --backend tuned --experiment synthetic-001
python scripts/robustness.py --backend tuned --experiment synthetic-001
python scripts/analyze_errors.py --backend tuned --experiment synthetic-001
python scripts/report.py --experiment synthetic-001
```

Não sobrescrever experiências; usar synthetic-002 para outra configuração.
Manifest local regista model ID/revision resolvida, hashes dos pesos locais,
configs, dataset version/source, seed, tamanhos, bibliotecas, timestamp e métricas.
O pipeline recusa adapters de outro dataset ou pesos locais alterados.
Para testar outra arquitetura, rever também módulos LoRA, tokenizer e exportação.
Verificar comprimento dos prompts sem truncar a taxonomia/resposta.

## Avaliação e limites científicos

Baselines: --backend ml (TF-IDF char + Logistic Regression, fit só no treino),
--backend base (HF Base) e --backend ollama-base (qwen3:0.6b com template nativo,
thinking=false e schema). O Ollama instalado não substitui os pesos HF de treino.
Accuracy, macro/weighted F1, precision/recall/F1/suporte por categoria, matriz,
outputs inválidos, erros e latência média/mediana/p95 são guardados localmente.
Relatório em reports/<experiment>/evaluation.md exclui métricas de outro dataset.

A tarefa devolve apenas {"category":"<categoria válida>"}. O parser rejeita
JSON inválido, chaves duplicadas, texto extra, confidence e categorias inexistentes.
Prompts não incluem group, counterparty, template_family ou entry_group como
features separadas; o texto pode naturalmente mencionar a contraparte fictícia.
Sensibilidade a direção/montante não equivale a erro: ablações podem mudar a
classificação correta. Testes finais só depois de congelar candidato:

```powershell
python scripts/evaluate.py --backend base --split test --final --experiment synthetic-001
python scripts/evaluate.py --backend tuned --split test --final --experiment synthetic-001
python scripts/evaluate.py --backend base --split hard_test --final --experiment synthetic-001
python scripts/evaluate.py --backend tuned --split hard_test --final --experiment synthetic-001
```

Este corpus é equilibrado e baseado em templates/regras. Há pistas de categoria
mais explícitas do que em muitos extratos reais. Classes patrimoniais, prendas,
compras residuais e reembolsos exigem contexto que pode faltar em produção.
**Accuracy sintética elevada não comprova qualidade nos movimentos reais.**
Não usar teste para tuning. Integração futura exigirá avaliação real privada,
separada e explicitamente autorizada; o projeto atual não a faz.

## Exportação, quantização e Ollama

```powershell
python scripts/export_model.py --experiment synthetic-001
python C:\tools\llama.cpp\convert_hf_to_gguf.py models/synthetic-001/merged --outtype f16 --outfile models/synthetic-001/denarius-f16.gguf
C:\tools\llama.cpp\build\bin\Release\llama-quantize.exe models/synthetic-001/denarius-f16.gguf models/synthetic-001/denarius-q4_k_m.gguf Q4_K_M
ollama create denarius-movement -f configs/Modelfile.example
```

Merge em CPU FP32, nunca sobre pesos 4-bit. Confirmar suporte Qwen3 no checkout
llama.cpp usado, registar versão/commit e testar pesos efetivamente produzidos.
Modelfile literal; inferência treinada usa raw=true, temperature=0 e JSON Schema.
Conversão GGUF e ollama create são comandos externos, não automáticos no notebook.

```python
from denarius_model.config import dataset_path, read_json
from denarius_model.inference import OllamaClassifier, classify
cats = read_json(dataset_path('processed/categories.json'))
classify(OllamaClassifier(), {'description':'ALDI compras supermercado SYN-LAROMA',
    'amount':'37.25','direction':'debito_contabilistico'}, cats)
```

Criar dois modelos Ollama F16/Q4 e comparar no mesmo template/schema para isolar
quantização. HF livre versus Ollama constrained mistura efeitos de decoding.
Benchmark quente/frio: scripts/benchmark_ollama.py --model denarius-movement.
VRAM PyTorch não mede Ollama; nvidia-smi mede uso total da GPU.
Treino completo, exportação e benchmark final não foram executados automaticamente.

## Estrutura e continuidade

src/denarius_model: módulos; scripts: comandos; configs: modelo, treino, dataset e
taxonomia; notebooks: orquestração; tests: testes sintéticos; data/synthetic: corpus
público. models/outputs/reports e dados antigos: apenas artefactos locais privados.

AGENTS.md orienta futuras sessões. Skills em .agents/skills: denarius-dataset,
denarius-experiments e denarius-ollama. Guardam regras e contexto para retomar;
não exigem voltar a consultar movimentos reais ou o container. O corpus pode
ser regenerado só com os ficheiros do repositório.

## Fontes técnicas

- [Qwen3 Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [PEFT](https://huggingface.co/docs/peft/index)
- [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Conversor llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
- [Ollama](https://github.com/ollama/ollama/blob/main/docs/api.md)

## Licença

Distribuído sob a [licença MIT](LICENSE). Copyright © 2026 Denarius Decision Model contributors.
