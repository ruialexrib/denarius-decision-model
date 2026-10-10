---
name: denarius-ollama
description: Exportar o modelo Denarius treinado no corpus sintético para merge LoRA, GGUF, Q4_K_M e Ollama local, avaliando quantização e latência.
---

# Exportação e Ollama

Ler README.md, export.py, inference.py, configs/Modelfile.example e manifest da
experiência sintética. Corpus ativo data/synthetic; nenhum leitor Docker/SQL.
Não reutilizar adapters dos dados reais antigos. Dataset/source e fingerprint
da base devem corresponder ao manifest antes de merge. Usar IDs synthetic-*.

scripts/export_model.py --experiment synthetic-001 faz merge CPU FP32 e grava
safetensors locais. Não fazer merge em pesos 4-bit. Converter com llama.cpp
convert_hf_to_gguf.py para F16, depois llama-quantize Q4_K_M. Confirmar suporte
Qwen3 no checkout usado; registar commit llama.cpp e versão Ollama localmente.
Conversão/ollama create são passos externos, não executados pelo notebook.

Modelfile aponta por omissão para models/synthetic-001/denarius-q4_k_m.gguf;
adaptar se outro ID. Template literal e temperature=0. Criar denarius-movement
local, sem Hub/push/publish/releases. Modelos/adapter/GGUF ficam ignorados mesmo
que treino use dados sintéticos. Não alterar aplicação DenariusAI.

Modelo treinado usa completion prompt HF e raw=true. Baseline qwen3:0.6b usa
template nativo/thinking=false. JSON Schema enum com as 70 categorias existentes
+ parser estrito. Só loopback; nunca endpoint cloud. Testar primeiro registo
sintético e avaliar validação com --backend ollama. Testes finais só após
congelamento; são sintéticos e não demonstram qualidade em movimentos reais.

Comparar F16/Q4 no mesmo Ollama/template/schema para isolar quantização; comparar
HF livre com Ollama constrained mistura decoding com quantização.
scripts/benchmark_ollama.py --model <nome> mede frio/quente na validação;
consultar média/mediana/p95, accuracy e tokens/s. VRAM nvidia-smi é total da GPU,
não processo isolado. Descarregar Ollama da GPU antes de treino HF se necessário.

Não afirmar conversão completa por haver comandos/Modelfile. Confirmar artefactos
produzidos e inferência real local. No final distinguir passos preparados dos
executados, declarar limites do corpus sintético e executar auditoria Git.
