# Corpus financeiro sintético v1

**Todos os movimentos são inventados.** Não contém movimentos extraídos,
identificadores pessoais, IBAN, contas ou montantes copiados de pessoas reais.
As 70 categorias ativas e os grupos correspondem à taxonomia existente,
congelada em `configs/taxonomy.yaml` por pedido do utilizador. Categorias
inativas não integram a classificação. Não foram criadas categorias alternativas.

Uma revisão estrutural local única informou formatos como texto livre,
PAGAMENTO, TRF MB, LEVANTAMENTO e ORDENADO. Números, pessoas e descrições
individuais de origem não foram copiados. Intervalos de montante e contextos
foram escritos manualmente. O gerador não precisa de Docker, SQL, Ollama ou
dados reais. Marcas públicas presentes nas categorias (ALDI, Netflix, etc.)
continuam nos contextos para as distinguir. Contrapartes `SYN-*` são fictícias.

## Composição

- 8 400 exemplos de treino e 1 400 em cada split de validação, teste e teste difícil.
- 70 categorias em cada split, equilibradas por construção; não reproduz
  frequências de utilização reais.
- Variações de caixa, acentos, espaçamento, texto curto/contextual e referências
  explicitamente sintéticas. Montantes positivos com só Debit ou Credit não zero.
- Direção é débito/crédito **contabilístico**, não sinal de entrada/saída bancária.
- Estornos mantêm a categoria da despesa original; reembolsos de adiantamentos
  a terceiros pertencem à categoria de rendimentos Reembolsos.

## Labels e ambiguidades

Contrato: `{"category":"<grupo / categoria existente>"}`. Grupo integra o label
para desambiguar; nunca é feature de entrada. Contextos contêm pistas explícitas:

- Renda é rendimento de arrendamento, não pagamento de renda da habitação.
- IRS anual distingue-se de retenção; IUC, IMI e selo são explicitados.
- Formação refere cursos/workshops; propinas referem ensino superior; material
  escolar refere artigos e não serviços educativos.
- Reparação doméstica pontual difere de obra/remodelação estrutural.
- Constituição de poupanças é verba programada para formar reserva; contas
  patrimoniais descrevem lançamentos de ativos/contrapartidas explicitamente.
- Dinheiro patrimonial distingue-se de reforço de caixa/fundo de maneio.
- Prenda exige finalidade de oferta. Compras diversas são rotineiras; compras
  extraordinárias são residuais, não recorrentes e explicitamente identificadas.

Se textos reais não contiverem estas pistas, o label pode ser ambíguo. O corpus
não resolve informação ausente e não valida a interpretação contabilística de
um lançamento real. Rever estas convenções antes de qualquer integração.

## Splits

Um cenário é contraparte/contexto de uma categoria. Variantes ficam no mesmo
split; descrições canónicas não atravessam splits. Test normal pode partilhar
contrapartes com treino, mas usa outros cenários. Hard usa contrapartes fictícias,
contextos parafraseados e famílias de template ausentes nos outros splits.
Marcas que definem categorias continuam conhecidas; não afirmar que essas
cadeias comerciais são inéditas. O vocabulário e significado podem sobrepor-se.
Esta separação não prova generalização no mundo real.

## Reprodução e Git

`python scripts/generate_dataset.py` recria raw, categorias, estatísticas, splits
e manifest com seed fixa e hashes. `prepare_dataset.py` regenera o mesmo corpus.
`python scripts/audit_git.py` compara os oito ficheiros públicos byte a byte com
o gerador. Alterações manuais ao dataset exigem rever o gerador.
Os dados antigos em `data/raw`, `data/processed` e `data/splits` permanecem
privados, ignorados e fora do pipeline. Só os JSON/JSONL explicitamente permitidos
no `.gitignore` são públicos. Modelos, outputs e relatórios continuam ignorados.

Dataset sob a licença MIT do repositório. Métricas deste corpus artificial
não demonstram qualidade em movimentos reais do DenariusAI.
