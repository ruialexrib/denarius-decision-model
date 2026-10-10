---
name: denarius-dataset
description: Gerar, rever e preparar o corpus sintético Denarius com as categorias existentes, proveniência reproduzível e separação de cenários entre splits.
---

# Corpus sintético Denarius

Ler README.md, configs/taxonomy.yaml, configs/dataset.yaml,
src/denarius_model/synthetic.py e data/synthetic/README.md. O fluxo foi mudado
por pedido do utilizador: não há scripts de extração SQL/Docker. Não consultar
novamente containers ou movimentos privados para gerar exemplos.

## Contexto que deve persistir

A taxonomia foi lida uma vez da instalação: 70 categorias ativas com label
exato grupo / categoria. Há categorias sem movimentos previamente classificados;
todas têm contextos sintéticos. Categorias inativas foram excluídas. Não inventar
outros labels nem simplificar silenciosamente os nomes existentes.

Uma revisão estrutural única dos movimentos locais informou texto livre,
PAGAMENTO, TRF MB, LEVANTAMENTO, ORDENADO e marcas das categorias. Não copiámos
registos, pessoas, referências, IDs ou montantes. Intervalos e contextos foram
escritos manualmente. Os futuros dados vêm apenas deste código/taxonomia.
Marcas como ALDI/Netflix/ChatGPT são referências públicas necessárias aos labels.
Contrapartes SYN-* são fictícias. Não guardar dados privados nas skills.

Direção continua contabilística: debito_contabilistico/credito_contabilistico.
Não trocar para sinal bancário. Renda é rendimento de arrendamento. Conta à
Ordem/Conta Poupança/Dinheiro/Investimentos exigem contexto patrimonial explícito;
Constituição de Poupanças é verba programada para formar reserva. Estorno mantém
categoria da despesa original; Reembolsos refere despesas adiantadas a terceiros.
Ambiguidades de IRS/propinas/formação/obras/compras residuais constam na ficha do
corpus. Não atribuir labels sem uma pista suficiente no contexto sintético.

## Geração e splits

python scripts/generate_dataset.py, ou prepare_dataset.py, gera apenas
 data/synthetic. Dataset v1: 120 train + 20 validation + 20 test + 20 hard_test
por categoria; 12 600 exemplos, seed 42. As contagens atuais prevalecem em config.
Não usar data/raw, data/processed ou data/splits antigos: são privados e ignorados.

Cenário = categoria + contraparte + contexto. Variantes permanecem num split;
não há overlap de descrições canónicas. Test normal pode conhecer contrapartes,
mas reserva cenários. Hard tem contrapartes fictícias, contextos parafraseados e
famílias de template novos. Marcas definidoras de classes permanecem conhecidas;
não afirmar que todas as cadeias comerciais são inéditas. Sobreposição semântica
é inevitável e a avaliação é artificial. Corpus equilibrado não representa
frequências reais e não demonstra generalização no DenariusAI.

Para aumentar volume, ampliar diversidade dos contextos/templates. O gerador
rejeita contagens superiores à diversidade suportada. Não duplicar texto só
mudando montante. Após alteração, regenerar artefactos/manifest e executar testes;
o dataset_version considera raw, exemplos preparados, taxonomia e configuração.
Alterações semânticas exigem atualizar generator_version no código e config.

## Git

Oito ficheiros JSON/JSONL e README do corpus são explicitamente permitidos no
.gitignore. Outros dados, modelos, outputs e relatórios continuam ignorados.
python scripts/audit_git.py regenera o corpus e compara byte a byte; marcador
synthetic=true não basta para permitir um ficheiro alterado manualmente.
Rever código de geração e labels antes do commit; não usar git add -f.
Executar testes de deterministicidade, cobertura, montantes, duplicates, cenários,
contrapartes/templates novos e proteção dos antigos datasets privados.
