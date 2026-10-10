import re
import unicodedata
from decimal import Decimal, InvalidOperation

def redact(text):
    text = unicodedata.normalize('NFKC', str(text))
    for pattern in [r'\b[A-Z]{2}\s*\d{2}(?:\s*[A-Z0-9]){11,30}\b',
                    r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', r'https?://\S+',
                    r'\b\d[\d .-]{6,}\d\b']:
        text = re.sub(pattern, '[REDACTED]', text, flags=re.I)
    return ' '.join(text.upper().split())

def normalize(row):
    try:
        debit, credit = Decimal(str(row['debit'])), Decimal(str(row['credit']))
    except InvalidOperation as exc:
        raise ValueError('Montante inválido') from exc
    if not debit.is_finite() or not credit.is_finite() or min(debit, credit)<0 or ((debit==0)==(credit==0)):
        raise ValueError('Montante inválido')
    description = redact(row['description'])
    category = str(row['category']).strip()
    if not description or not category:
        raise ValueError('Texto ou categoria vazio')
    return dict(description=description, amount=str(max(debit,credit)),
                direction='debito_contabilistico' if debit else 'credito_contabilistico',
                category=category, entry_group=row['entry_group'])

def pattern(text):
    text = unicodedata.normalize('NFKD', text)
    return ' '.join(re.sub(r'[^A-Z ]', ' ', text.upper()).split())
