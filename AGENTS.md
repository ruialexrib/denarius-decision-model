# Continuidade do Denarius Decision Model

Projeto de investigação local independente. Responder em português. Fluxo atual:
**dataset sintético versionável com as categorias ativas existentes**, sem Docker
ou SQL. Ler README.md, configs e a skill aplicável antes de alterar o pipeline.

## Regras permanentes

- Dados financeiros reais, secrets, modelos, adapters, checkpoints e GGUF nunca
  vão para GitHub, Hub, tracking ou cloud. Não publicar ou criar releases por iniciativa própria.
- Apenas o corpus sintético explicitamente permitido em data/synthetic pode ser
  versionado; comparar com o gerador através de scripts/audit_git.py.
- Não consultar novamente a instalação DenariusAI ou movimentos privados para
  regenerar dados. Fonte futura: configs/taxonomy.yaml + synthetic.py + dataset.yaml.
  Nova utilização de dados reais depende de um pedido explícito do utilizador.
- Não alterar DenariusAI ou base de dados. Leitores Docker/SQL foram removidos.
- Notebooks no Git sem outputs; testes usam só dados sintéticos.
- Testes finais só depois de congelar candidato. Não usar test/hard_test para tuning.
- Métricas sintéticas não demonstram qualidade em movimentos reais ou prontidão
  para integração. Preservar esta limitação nos relatórios e conclusões.

## Skills de continuidade

- .agents/skills/denarius-dataset/SKILL.md: origem sintética, taxonomia e splits.
- .agents/skills/denarius-experiments/SKILL.md: baselines, SFT, avaliação e limites.
- .agents/skills/denarius-ollama/SKILL.md: merge, GGUF e benchmark local.

As skills guardam as decisões reutilizáveis. Não gravar exemplos privados nelas.
Consultar apenas a skill necessária; não é preciso redescobrir o schema SQL.
Verificar código/configs atuais se uma nota histórica divergir.

## Retomar

Ler git status, configs/dataset.yaml e data/synthetic/manifest.json. Verificar
pesos HF via local_path e CUDA; qwen3:0.6b no Ollama é baseline adicional, não
substitui pesos de treino. Novas experiências usam synthetic-001, synthetic-002,
etc.; não reutilizar modelos/relatórios das experiências reais antigas.
Antes de repetir treino, consultar outputs/<experiência sintética>/experiment.json;
validar dataset version/source e hashes do modelo. Não ler os raw reais antigos.

Comandos por scripts ou CLI instalada denarius-model. Notebook pode executar
todo o treino/validação; não executar também scripts sobre o mesmo ID.
Depois de atualizar notebook/código, reabrir o notebook e reiniciar kernel para
não executar funções antigas que ainda possam apontar para datasets privados.

## Verificações

python -m pytest -q; python scripts/audit_git.py; rever git ls-files e diff staged.
A auditoria valida regeneração e paths, não substitui revisão de conteúdo do
código/docs. Nunca git add -f sobre dados privados. Alterações ao dataset exigem
testes de deterministicidade, cobertura e overlap e regeneration dos oito
artefactos públicos; preserve LF via .gitattributes.
Se afetar treino, usar scripts/smoke_lora.py como teste sintético CUDA/PEFT;
não confundir esse smoke com treino completo ou medição de VRAM do 0.6B.
