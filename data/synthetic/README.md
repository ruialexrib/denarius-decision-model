# Synthetic financial corpus v1

**All transactions are invented.** The corpus contains no extracted transactions,
personal identifiers, IBANs, account numbers or amounts copied from real people.
The 70 active categories and their groups match the existing taxonomy, frozen in
`configs/taxonomy.yaml` at the user's request. Inactive categories are excluded
from classification. No alternative categories were created. Category labels and
transaction descriptions retain their original Portuguese task language.

A one-time local structural review informed formats such as free text,
PAGAMENTO, TRF MB, LEVANTAMENTO and ORDENADO. Source numbers, people and individual
descriptions were not copied. Amount ranges and contexts were written manually.
The generator requires no Docker, SQL, Ollama or real data. Public brands present
in category labels, such as ALDI and Netflix, remain in the contexts to distinguish
those categories. Counterparties prefixed with `SYN-*` are fictional.

## Composition

- 8,400 training examples and 1,400 examples in each validation, test and hard-test split.
- 70 categories in every split, balanced by construction; real usage frequencies
  are not reproduced.
- Variations in capitalization, accents, spacing, short/contextual descriptions
  and explicitly synthetic references. Amounts are positive, with only Debit or
  Credit nonzero.
- Direction represents **accounting** debit/credit, not bank inflow/outflow signs.
- Expense reversals retain the original expense category; reimbursements of
  advances to third parties belong to the income category `Reembolsos`.

## Labels and ambiguities

Output contract: `{"category":"<existing group / category>"}`. The group forms part
of the label to disambiguate categories; it is never an input feature. Contexts
contain explicit cues:

- `Renda` represents rental income, not a housing rent payment.
- Annual IRS income tax is distinguished from withholding; IUC vehicle tax,
  IMI property tax and stamp duty are explicitly identified.
- `Formação` covers courses/workshops; `Propinas` covers higher education tuition;
  school supplies represent items rather than educational services.
- Occasional household repairs differ from structural renovation work.
- Savings contributions represent scheduled funds for building a reserve;
  asset accounts explicitly describe asset entries and their counterparts.
- Cash held as an asset is distinguished from replenishing petty cash or
  working capital.
- Gifts require an explicit gift purpose. Miscellaneous purchases are routine;
  extraordinary purchases are residual, nonrecurring and explicitly identified.

When real descriptions lack these cues, labels may be ambiguous. The corpus does
not resolve missing information or validate the accounting interpretation of a
real entry. Review these conventions before any integration.

## Splits

A scenario combines a counterparty and context within a category. Variants remain
in the same split; canonical descriptions do not cross split boundaries. The
standard test may share counterparties with training but uses different scenarios.
The hard test uses fictional counterparties, paraphrased contexts and template
families absent from the other splits. Brands defining categories remain known;
do not claim that those retail chains are unseen. Vocabulary and meaning may
overlap. This separation does not prove real-world generalization.

## Reproducibility and Git

`python scripts/generate_dataset.py` recreates raw data, categories, statistics,
splits and the manifest using a fixed seed and hashes. `prepare_dataset.py`
regenerates the same corpus. `python scripts/audit_git.py` compares the eight public
data files against the generator byte for byte. Manual dataset changes require
corresponding generator changes.

Earlier data under `data/raw`, `data/processed` and `data/splits` remains private,
ignored and outside the pipeline. Only JSON/JSONL files explicitly allowed by
`.gitignore` are public. Models, outputs and generated reports remain ignored.

The dataset is distributed under the repository's MIT License. Metrics on this
artificial corpus do not demonstrate quality on real DenariusAI transactions.
