"""Corpus inventado, sem consulta de Docker, base de dados ou ficheiros privados."""
import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from . import ROOT
from .preprocessing import normalize, pattern

GENERATOR_VERSION = 'synthetic-v1'
SYNTHETIC_ROOT = ROOT / 'data/synthetic'
SPLITS = ('train', 'validation', 'test', 'hard_test')

# Vocabulário escrito manualmente. As marcas públicas só são usadas quando
# necessárias para distinguir categorias que já têm essas marcas no nome.
# Montantes são intervalos inventados, não estatísticas de movimentos reais.
# Cada entrada: limite inferior/superior, quatro contextos normais, dois novos.
CONTEXTS = {
    'Água': (8, 110, ['fatura abastecimento de água', 'consumo mensal de água e saneamento', 'serviço de águas municipais', 'pagamento fornecimento água'], ['liquidação serviço hídrico doméstico', 'tarifa saneamento e abastecimento hídrico']),
    'Caixas e Fundo de Maneio': (10, 350, ['levantamento para fundo de maneio', 'reposição caixa operacional', 'adiantamento pequeno caixa', 'reforço caixa de despesas correntes'], ['numerário para pequenas despesas da caixa', 'aprovisionamento do fundo operacional em dinheiro']),
    'Combustível': (5, 180, ['abastecimento gasóleo', 'compra gasolina sem chumbo', 'pagamento posto combustível', 'reabastecimento automóvel GPL'], ['tanque atestado com diesel', 'aquisição de carburante para viatura']),
    'Comissões e outras despesas bancárias': (0.5, 70, ['comissão manutenção conta', 'taxa transferência interbancária', 'anuidade cartão bancário', 'custo processamento operação bancária'], ['encargo de gestão do pacote bancário', 'tarifário cobrado pelo banco por transferência']),
    'Compras Diversas': (2, 200, ['compra habitual de pequenas utilidades', 'artigos correntes para uso diário', 'aquisição rotineira de acessórios diversos', 'consumíveis de uso habitual'], ['pequenos artigos de utilização quotidiana', 'reposição de utilidades correntes sem rubrica específica']),
    'Compras Material Escolar': (1, 160, ['compra cadernos e lápis escolares', 'aquisição estojo e mochila escolar', 'material para sala de aula', 'papelaria para aulas'], ['consumíveis escolares para o período letivo', 'réguas e fichas de apoio às aulas']),
    'Compras na Farmácia': (1, 200, ['medicamentos na farmácia', 'compra receita farmacêutica', 'aquisição fármacos prescritos', 'produtos farmacêuticos'], ['dispensa de medicação prescrita', 'levantamento de comprimidos com receita']),
    'Compras na Feira': (1, 140, ['compras na feira semanal', 'aquisição banca da feira', 'pagamento feirante mercado ao ar livre', 'produtos da feira local'], ['compra no mercado ambulante', 'produtos adquiridos numa banca itinerante']),
    'Compras na Mercearia': (1, 140, ['compra mercearia de bairro', 'alimentos no pequeno comércio alimentar', 'compras loja de mercearia', 'mercearia tradicional alimentos'], ['cabaz de víveres no comércio alimentar de proximidade', 'produtos de despensa numa venda de bairro']),
    'Compras no ALDI': (1, 250, ['ALDI compras supermercado', 'pagamento cartão ALDI', 'ALDI cabaz alimentar', 'compra semanal ALDI'], ['Aldi aquisição de víveres', 'A L D I compra de produtos de despensa']),
    'Compras no Auchan': (1, 300, ['AUCHAN compras hipermercado', 'pagamento cartão AUCHAN', 'AUCHAN cabaz alimentar', 'compra semanal AUCHAN'], ['Auchan aquisição de víveres', 'AUCHAN DRIVE recolha do cabaz alimentar']),
    'Compras no Continente': (1, 300, ['CONTINENTE compras supermercado', 'pagamento CONTINENTE BOM DIA', 'CONTINENTE cabaz alimentar', 'compra semanal CONTINENTE'], ['Continente Modelo aquisição de víveres', 'CONTINENTE ONLINE encomenda de produtos de despensa']),
    'Compras no LIDL': (1, 250, ['LIDL compras supermercado', 'pagamento cartão LIDL', 'LIDL cabaz alimentar', 'compra semanal LIDL'], ['Lidl aquisição de víveres', 'L I D L compra de produtos de despensa']),
    'Compras no Mercadona': (1, 250, ['MERCADONA compras supermercado', 'pagamento cartão MERCADONA', 'MERCADONA cabaz alimentar', 'compra semanal MERCADONA'], ['Mercadona aquisição de víveres', 'MERCADONA encomenda de produtos de despensa']),
    'Compras no Pingo Doce': (1, 250, ['PINGO DOCE compras supermercado', 'pagamento cartão PINGO DOCE', 'PINGO DOCE cabaz alimentar', 'compra semanal PINGO DOCE'], ['Pingo-Doce aquisição de víveres', 'P. DOCE compra de produtos de despensa']),
    'Compras no Talho': (1, 180, ['compra carne no talho', 'pagamento talho bovino e aves', 'aquisição carne fresca talho', 'talho encomenda de carnes'], ['produtos cárneos num açougue', 'cortes de carne de um comerciante especializado']),
    'Condomínio': (10, 350, ['quota mensal condomínio', 'encargo administração condomínio', 'pagamento condomínio prédio', 'contribuição manutenção áreas comuns condomínio'], ['rateio das despesas comuns do edifício', 'quotização dos condóminos para a administração do prédio']),
    'Cuidados com Animais': (2, 600, ['consulta veterinária', 'compra ração animal', 'vacinação animal de companhia', 'tratamento clínica veterinária'], ['assistência médica ao animal doméstico', 'alimentos e acessórios para animal de companhia']),
    'Cuidados de Beleza': (5, 200, ['serviço cabeleireiro', 'manicure em salão', 'tratamento estético facial', 'corte e coloração cabelo'], ['serviço de barbeiro e penteado', 'pedicure e serviço de estética']),
    'Cuidados Médicos': (10, 1200, ['consulta médica clínica', 'exames de diagnóstico', 'tratamento dentário', 'sessão fisioterapia'], ['honorários de assistência clínica', 'análises laboratoriais de saúde']),
    'Despesas com cuidados pessoais': (1, 140, ['compra shampoo e higiene pessoal', 'produtos de higiene corporal', 'aquisição sabonete e pasta dentífrica', 'consumíveis higiene diária'], ['reposição de desodorizante e gel de banho', 'artigos de limpeza corporal para uso pessoal']),
    'Despesas com Estudo e Formação': (5, 1300, ['curso de formação profissional', 'inscrição workshop educativo', 'compra curso online', 'sessões de explicações'], ['ação de capacitação profissional', 'oficina de aprendizagem e aperfeiçoamento']),
    'Eletricidade': (10, 250, ['fatura eletricidade', 'consumo energia elétrica', 'pagamento luz habitação', 'fornecimento eletricidade doméstica'], ['liquidação dos quilowatt-hora domésticos', 'tarifa de energia elétrica da residência']),
    'Estacionamento e Portagens': (0.5, 100, ['pagamento portagem autoestrada', 'parque estacionamento', 'tarifa parquímetro', 'cobrança passagem via com portagem'], ['taxa de circulação numa via concessionada', 'permanência em silo automóvel']),
    'Hipoteca da Habitação': (100, 2200, ['prestação crédito habitação', 'pagamento empréstimo hipotecário', 'mensalidade financiamento da casa', 'amortização hipoteca habitação'], ['liquidação de capital do crédito da residência', 'encargo mensal do mútuo hipotecário residencial']),
    'IMI - Imposto Municipal sobre Imóveis': (10, 1500, ['pagamento IMI', 'imposto municipal sobre imóveis', 'prestação IMI imóvel', 'liquidação IMI prédio'], ['tributação municipal anual do prédio', 'cobrança I.M.I. da propriedade imobiliária']),
    'Impostos Diversos': (1, 1600, ['pagamento imposto do selo', 'liquidação taxa fiscal diversa', 'tributo não IRS IMI ou IUC', 'encargo fiscal selo'], ['cobrança de tributo residual sobre um ato', 'liquidação de imposto de selo numa operação']),
    'Inspeção Automóvel': (10, 140, ['inspeção periódica automóvel', 'pagamento centro inspeções viatura', 'controlo técnico obrigatório carro', 'reinspeção automóvel'], ['verificação periódica obrigatória do veículo', 'serviço de inspeção técnica da viatura']),
    'IRS - Imposto sobre o Rendimento das Pessoas Singulares': (10, 6500, ['liquidação anual IRS', 'pagamento acerto declaração IRS', 'nota cobrança IRS anual', 'imposto rendimento pessoas singulares liquidação'], ['saldo a pagar da declaração anual de rendimentos', 'cobrança da liquidação final de I.R.S.']),
    'IRS - Retenção': (1, 1600, ['retenção IRS na fonte', 'desconto IRS sobre vencimento', 'retenção fiscal mensal salário', 'IRS retido pagamento honorários'], ['parcela de I.R.S. retida antes do pagamento', 'desconto mensal do imposto sobre o ordenado']),
    'IUC - Imposto Único Circulação': (10, 850, ['pagamento IUC', 'imposto único circulação', 'tributo anual circulação veículo', 'liquidação IUC automóvel'], ['cobrança do selo anual da viatura', 'imposto anual pela circulação de veículo']),
    'Mesadas e Semanadas': (2, 300, ['transferência mesada familiar', 'pagamento semanada', 'atribuição mesada mensal', 'apoio semanal em mesada'], ['entrega periódica de dinheiro de bolso', 'verba semanal para despesas pessoais em semanada']),
    'Netflix': (3, 40, ['NETFLIX assinatura mensal', 'pagamento subscrição NETFLIX', 'NETFLIX plano streaming', 'mensalidade NETFLIX entretenimento'], ['Netflix renovação do serviço de vídeo', 'NETFLIX.COM acesso ao catálogo de séries']),
    'Piscina e Ginásio': (2, 200, ['mensalidade ginásio', 'entrada piscina', 'quota clube fitness', 'aulas de natação'], ['acesso ao centro de exercício físico', 'sessão de treino e utilização de piscina']),
    'Prestação Empréstimo Carro': (40, 1500, ['prestação crédito automóvel', 'financiamento carro mensalidade', 'amortização empréstimo viatura', 'pagamento crédito do carro'], ['liquidação mensal do financiamento de veículo', 'encargo do mútuo para aquisição da viatura']),
    'Propinas': (20, 3500, ['propina universidade', 'mensalidade propinas curso superior', 'pagamento propina faculdade', 'prestação propinas ensino superior'], ['encargo de matrícula curricular universitária', 'anuidade académica relativa às propinas']),
    'Quotas para OCC -  Ordem Contabilistas Certificados': (5, 400, ['quota OCC ordem contabilistas certificados', 'pagamento quotização OCC', 'OCC quota profissional', 'mensalidade ordem contabilistas certificados'], ['quotização da Ordem dos Contabilistas Certificados', 'O.C.C. contribuição profissional periódica']),
    'Seguro Acidentes de Trabalho': (5, 1800, ['prémio seguro acidentes trabalho', 'apólice acidente laboral', 'seguro proteção acidentes no trabalho', 'renovação seguro acidentes profissionais'], ['cobertura seguradora de sinistros laborais', 'prémio da proteção contra acidente de trabalho']),
    'Seguro Automóvel': (10, 1500, ['prémio seguro automóvel', 'renovação apólice carro', 'cobertura seguradora viatura', 'seguro responsabilidade civil automóvel'], ['prémio da proteção seguradora do veículo', 'seguro contra danos e sinistros rodoviários']),
    'Seguro Casa': (5, 1500, ['prémio seguro multirriscos habitação', 'renovação seguro casa', 'apólice proteção imóvel residencial', 'seguro incêndio da habitação'], ['cobertura seguradora de danos na residência', 'prémio multirriscos do edifício habitacional']),
    'Seguro Cooperativa Portuense': (5, 1500, ['seguro Cooperativa Portuense prémio', 'renovação apólice Cooperativa Portuense', 'cobertura seguradora da Cooperativa Portuense', 'pagamento seguro cooperativo Portuense'], ['prémio segurador contratado pela Cooperativa Portuense', 'proteção cooperativa Portuense renovada']),
    'Seguro de Vida': (2, 1200, ['prémio seguro vida', 'renovação apólice de vida', 'seguro proteção falecimento', 'cobertura seguradora vida'], ['prémio da proteção seguradora por morte', 'apólice que cobre risco de vida']),
    'Televisão, Comunicações e Internet': (5, 250, ['fatura telecomunicações internet', 'pagamento pacote televisão internet telefone', 'mensalidade comunicações móveis', 'serviço banda larga doméstica'], ['encargo do acesso de fibra e televisão', 'liquidação dos serviços de voz e dados móveis']),
    'Transportes': (1, 250, ['bilhete metro', 'passe autocarro', 'viagem comboio', 'serviço táxi deslocação'], ['título de transporte ferroviário', 'deslocação urbana em transporte público']),
    'Compras de Eletrodomésticos': (10, 3000, ['compra frigorífico', 'aquisição máquina lavar roupa', 'compra forno elétrico', 'pagamento aspirador doméstico'], ['aquisição de aparelho doméstico para conservar alimentos', 'equipamento elétrico de lavagem para casa']),
    'Compras de Material Informático': (5, 4000, ['compra computador portátil', 'aquisição teclado e rato', 'compra monitor informático', 'pagamento componentes de computador'], ['hardware para estação de trabalho informática', 'equipamento de processamento e periféricos digitais']),
    'Compras de Móveis': (10, 6000, ['compra sofá', 'aquisição mesa e cadeiras', 'compra armário', 'pagamento mobiliário quarto'], ['peças de mobília para a sala', 'aquisição de estrutura de cama e roupeiro']),
    'Compras em Restaurantes': (2, 300, ['refeição restaurante', 'pagamento almoço restaurante', 'jantar em restaurante', 'serviço de refeição preparada'], ['consumo de refeição em estabelecimento de restauração', 'conta de mesa após refeição servida']),
    'Compras extraordinárias': (10, 4000, ['compra excecional não recorrente de artigo diverso', 'aquisição extraordinária pontual não classificada noutra rubrica', 'despesa não habitual com utilidade diversa', 'compra avulsa excecional fora das despesas habituais'], ['aquisição residual de caráter excecional', 'compra diversa fora da rotina e sem rubrica específica']),
    'Compras para Férias e Viagens': (5, 8000, ['reserva hotel férias', 'compra voo de férias', 'pacote agência viagens', 'alojamento turístico durante férias'], ['estadia turística e deslocação de lazer', 'reserva de viagem para período de descanso']),
    'Compras Vestuário': (1, 1800, ['compra roupa', 'aquisição calçado', 'pagamento vestuário', 'compra casaco e camisola'], ['peças de indumentária e sapatos', 'artigos têxteis para vestir']),
    'Constituição de Poupanças': (5, 5000, ['contribuição programada para constituir poupança', 'reforço planeado do objetivo de poupança', 'verba mensal destinada a formar reserva', 'constituição de novo fundo de poupança'], ['separação de rendimento para criar reserva futura', 'afetação programada de verba à formação de poupança']),
    'Manutenção da Casa': (5, 2200, ['reparação torneira avariada', 'manutenção caldeira doméstica', 'reparação fechadura da casa', 'serviço manutenção canalização'], ['conserto pontual de instalação doméstica', 'assistência de reparação sem remodelação do imóvel']),
    'Manutenção do Automóvel': (5, 3500, ['revisão oficina automóvel', 'troca pneus viatura', 'reparação travões carro', 'mudança óleo automóvel'], ['assistência mecânica à viatura', 'conserto de componentes do veículo na oficina']),
    'Obras e Remodelações': (50, 20000, ['obra remodelação cozinha', 'renovação integral casa banho', 'empreitada remodelação habitação', 'obras estruturais no imóvel'], ['reabilitação de divisão com empreitada de construção', 'intervenção de renovação e transformação do imóvel']),
    'Prendas': (1, 1800, ['compra presente aniversário', 'prenda para oferecer', 'aquisição lembrança para oferta', 'presente festivo para destinatário fictício'], ['artigo adquirido com finalidade de oferta', 'lembrança destinada a celebrar ocasião especial']),
    'Subscrição ChatGPT': (2, 350, ['OPENAI CHATGPT assinatura', 'subscrição CHATGPT plano pago', 'pagamento CHATGPT PLUS', 'CHATGPT renovação mensal'], ['OpenAI renovação do acesso ChatGPT', 'CHAT GPT adesão a plano premium']),
    'Subscrição LinkedIn': (2, 350, ['LINKEDIN PREMIUM subscrição', 'pagamento LINKEDIN plano profissional', 'LINKEDIN assinatura mensal', 'renovação LINKEDIN PREMIUM'], ['LinkedIn acesso premium para carreira', 'LINKED IN renovação da assinatura profissional']),
    'Conta à Ordem': (1, 8000, ['registo patrimonial saldo conta à ordem', 'movimento de tesouraria na conta corrente', 'regularização do saldo bancário à ordem', 'contrapartida patrimonial na conta à ordem'], ['movimento de ativo bancário imediatamente disponível', 'lançamento de património em depósito à vista']),
    'Conta Poupança': (1, 15000, ['registo patrimonial saldo conta poupança', 'movimento de ativo na conta poupança existente', 'regularização de depósito na conta poupança', 'contrapartida patrimonial da conta poupança'], ['lançamento do ativo num depósito de reserva existente', 'registo de património numa conta remunerada de poupança']),
    'Dinheiro': (1, 1800, ['registo patrimonial numerário disponível', 'movimento de ativo em dinheiro físico', 'acerto saldo numerário pessoal', 'contrapartida patrimonial da carteira de dinheiro'], ['lançamento de património em notas e moedas', 'registo do ativo em numerário detido']),
    'Investimentos': (5, 20000, ['registo patrimonial carteira investimentos', 'movimento ativo em títulos financeiros', 'contrapartida patrimonial de ações', 'regularização valor contabilístico de investimentos'], ['lançamento do ativo da carteira de instrumentos financeiros', 'registo de património em participações financeiras']),
    'Renda': (100, 2500, ['recebimento renda arrendamento', 'rendimento mensal de imóvel arrendado', 'crédito renda do inquilino fictício', 'receita de aluguer do imóvel'], ['receita periódica por cedência do uso de imóvel', 'rendimento de locação de propriedade']),
    'Salário': (200, 9000, ['crédito ordenado mensal', 'recebimento salário', 'vencimento trabalho dependente', 'transferência remuneração mensal'], ['abono de vencimento pelo empregador fictício', 'retribuição periódica de contrato de trabalho']),
    'Subsídios': (10, 4500, ['recebimento subsídio apoio social', 'crédito subsídio férias', 'pagamento subsídio natal', 'abono apoio por subsídio'], ['prestação de apoio social recebida', 'receita de abono subsidiado']),
    'Ajustes': (0.01, 2500, ['ajuste contabilístico positivo', 'regularização credora por diferença', 'acerto positivo saldo', 'correção credora de arredondamento'], ['lançamento de diferença favorável na conciliação', 'regularização positiva de valor contabilístico']),
    'Juros': (0.01, 700, ['recebimento juros depósito', 'crédito remuneração juros', 'juros vencidos aplicação financeira', 'rendimento de juros bancários'], ['provento por remuneração do capital depositado', 'crédito de rendimento financeiro por juros']),
    'Prémios': (5, 15000, ['recebimento prémio desempenho', 'crédito prémio concurso', 'bónus atribuído por mérito', 'pagamento prémio extraordinário'], ['gratificação recebida por resultado alcançado', 'receita de recompensa por desempenho']),
    'Reembolsos': (0.5, 3000, ['reembolso de despesa adiantada por conta de terceiro', 'restituição de verba antecipada a entidade fictícia', 'crédito reembolso adiantamento documentado', 'devolução de montante suportado por conta alheia'], ['receita de reposição de um adiantamento por terceiro', 'recuperação de verba paga em nome de outra entidade']),
    'Trabalhos ocasionais': (5, 5000, ['recebimento trabalho pontual independente', 'honorários serviço ocasional', 'receita tarefa avulsa prestada', 'pagamento trabalho extra não recorrente'], ['remuneração de prestação esporádica por conta própria', 'receita por execução de serviço avulso']),
}

NORMAL_TEMPLATES = (
    '{context} - {party}', '{party} / {context}',
    'PAGAMENTO {context} {party}', 'TRF MB {party} {context}',
    'MOVIMENTO {party}: {context}', '{context}; entidade {party}',
    'SEPA {context} - {party}', '{party} | {context} | REF [SINTETICA]',
)
HARD_TEMPLATES = (
    'REG. {party} // {context} // documento [SINTETICO]',
    '{context} (contraparte {party}) processamento eletrónico',
    'OP. {party} / detalhe: {context} / lote [SINTETICO]',
    'LANÇ. {context} — {party} — referência [SINTETICA]',
    'PROCESSAMENTO DIGITAL de {context}, entidade {party}',
    'DETALHE {context} / balcão virtual {party}',
)

def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode('utf-8')

def corpus_bytes(rows):
    return b''.join(canonical_bytes(row) for row in rows)

def array_bytes(rows):
    # One record per line: readable diffs, while remaining a standard JSON array.
    return b'[\n' + b',\n'.join(canonical_bytes(row).rstrip(b'\n') for row in rows) + b'\n]\n'

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def taxonomy():
    import yaml
    result = yaml.safe_load((ROOT/'configs/taxonomy.yaml').read_text(encoding='utf-8'))
    labels = result['categories']
    if not labels or len(labels) != len(set(labels)):
        raise ValueError('Taxonomia vazia ou com categorias duplicadas')
    for label in labels:
        if label.split(' / ', 1)[1] not in CONTEXTS:
            raise ValueError(f'Categoria sem contextos sintéticos: {label}')
    return labels

def dataset_config():
    import yaml
    cfg = yaml.safe_load((ROOT/'configs/dataset.yaml').read_text(encoding='utf-8'))
    if cfg['generator_version'] != GENERATOR_VERSION:
        raise ValueError('Versão do gerador não corresponde à configuração')
    if set(cfg['per_category']) != set(SPLITS) or any(type(v) is not int or v <= 0 for v in cfg['per_category'].values()):
        raise ValueError('Definir contagens positivas para os quatro splits')
    return cfg

def _parties(label, seed):
    # Pronunciáveis, fictícias e sem número que codifique a categoria.
    syllables = ['la','me','vi','ro','ca','nu','ta','fo','ze','pi','sa','do','be','lu','mo','ra']
    rng = random.Random(hashlib.sha256(f'{seed}:{label}:parties'.encode()).hexdigest())
    names = []
    while len(names) < 8:
        name = 'SYN-' + ''.join(rng.choice(syllables) for _ in range(5)).upper()
        if name not in names: names.append(name)
    return names

def _style(description, variant):
    mode = variant % 4
    if mode == 1: return description.upper()
    if mode == 2:
        return ''.join(c for c in unicodedata.normalize('NFKD',description) if not unicodedata.combining(c)).upper()
    if mode == 3: return description.replace(' ', '  ')
    return description

def generate_rows(labels=None, cfg=None):
    labels = taxonomy() if labels is None else labels
    cfg = dataset_config() if cfg is None else cfg
    rows = []
    for label in labels:
        name = label.split(' / ', 1)[1]
        low, high, contexts, hard_contexts = CONTEXTS[name]
        rng = random.Random(hashlib.sha256(f'{cfg["seed"]}:{label}'.encode()).hexdigest())
        parties = _parties(label, cfg['seed'])
        # A scene is an operation-context + counterparty pair. All its variants
        # stay in one split. Normal test knows parties, not these exact scenes.
        scenes = [(p, c) for p in parties[:6] for c in contexts]
        rng.shuffle(scenes)
        split_scenes = {'train':scenes[:16], 'validation':scenes[16:20], 'test':scenes[20:24],
                        'hard_test':[(p,c) for p in parties[6:] for c in hard_contexts]}
        for split in SPLITS:
            options = split_scenes[split]
            templates = HARD_TEMPLATES if split=='hard_test' else NORMAL_TEMPLATES
            limit = len(options)*len(templates)
            if cfg['per_category'][split] > limit:
                raise ValueError(f'Quantidade excede diversidade disponível: {label}/{split}')
            used = set(); count = 0; attempt = 0
            while count < cfg['per_category'][split]:
                party, context = options[attempt % len(options)]
                variant = attempt // len(options)
                template = templates[variant % len(templates)]
                text = template.format(context=context,party=party)
                # Income operations use a neutral receipt format, not an outgoing
                # payment. Expense reversals retain original category explicitly.
                income = label.startswith('Rendimentos ')
                text = text.replace('PAGAMENTO ', 'RECEBIMENTO ') if income else text
                refund = not income and not label.startswith('Património ') and attempt % 10 == 0
                if refund: text = 'ESTORNO DA OPERAÇÃO ORIGINAL: ' + text
                text = _style(text, count%4)
                signature = pattern(text)
                attempt += 1
                if signature in used: continue
                used.add(signature)
                # Fixed broad invented ranges; no reuse of real source amounts.
                cents = rng.randint(max(1,round(low*100)),round(high*100))
                credit = income or refund
                if label.startswith('Património '): credit = count % 2 == 1
                group = hashlib.sha256(f'{label}|{party}|{context}'.encode()).hexdigest()[:24]
                rows.append({'synthetic':True,'source_version':GENERATOR_VERSION,
                    'description':text,'debit':0 if credit else cents/100,
                    'credit':cents/100 if credit else 0,'category':label,
                    'entry_group':group,'split':split,'counterparty':party,
                    'template_family':('hard-' if split=='hard_test' else 'normal-')+str(variant%len(templates))})
                count += 1
    # Ordering independent of hash randomisation in the interpreter.
    rows.sort(key=lambda row:(SPLITS.index(row['split']),row['category'],row['entry_group'],row['description']))
    return rows

def public_payloads():
    labels = taxonomy(); cfg = dataset_config(); raw = generate_rows(labels,cfg)
    splits = {split:[] for split in SPLITS}
    for row in raw:
        normal = normalize(row)
        normal.update(synthetic=True,source_version=GENERATOR_VERSION,
                      counterparty=row['counterparty'],template_family=row['template_family'])
        splits[row['split']].append(normal)
    validate_splits(splits,labels)
    raw_bytes = corpus_bytes(raw)
    version = hashlib.sha256(raw_bytes+canonical_bytes(splits)+canonical_bytes(labels)+canonical_bytes(cfg)).hexdigest()
    stats = {'source':'synthetic','generator_version':GENERATOR_VERSION,'dataset_version':version,
        'seed':cfg['seed'],'categories':dict(Counter(r['category'] for r in raw)),
        'sizes':{s:len(v) for s,v in splits.items()},
        'split_categories':{s:dict(Counter(r['category'] for r in v)) for s,v in splits.items()},
        'invalid':0,'duplicates':0,'provenance':'Registos integralmente inventados; taxonomia existente e formatos informados por revisão estrutural local única. Sem cópias de movimentos ou montantes reais.'}
    payloads = {'raw/movements.jsonl':raw_bytes,
                'processed/categories.json':array_bytes(labels),
                'processed/statistics.json':canonical_bytes(stats)}
    payloads.update({f'splits/{s}.json':array_bytes(v) for s,v in splits.items()})
    manifest = {'source':'synthetic','generator_version':GENERATOR_VERSION,'dataset_version':version,
                'configuration':cfg,'files':{p:hashlib.sha256(b).hexdigest() for p,b in payloads.items()}}
    payloads['manifest.json'] = canonical_bytes(manifest)
    return payloads, stats

def validate_splits(splits, labels):
    seen_patterns=set(); seen_groups=set()
    for split,rows in splits.items():
        if not rows: raise ValueError(f'Split vazio: {split}')
        if {r['category'] for r in rows} != set(labels): raise ValueError(f'Faltam categorias em {split}')
        patterns = {pattern(r['description']) for r in rows}
        groups = {r['entry_group'] for r in rows}
        if seen_patterns & patterns: raise ValueError('Descrições canónicas atravessam splits')
        if seen_groups & groups: raise ValueError('Cenários atravessam splits')
        if len(patterns)!=len(rows): raise ValueError('Duplicados canónicos no split')
        seen_patterns.update(patterns); seen_groups.update(groups)
    known = {r['counterparty'] for s,rows in splits.items() if s!='hard_test' for r in rows}
    if known & {r['counterparty'] for r in splits['hard_test']}:
        raise ValueError('Contrapartes difíceis observadas no treino/validação/teste normal')
    normal_families={r['template_family'] for s,rows in splits.items() if s!='hard_test' for r in rows}
    if normal_families & {r['template_family'] for r in splits['hard_test']}:
        raise ValueError('Famílias de template difíceis não estão isoladas')

def generate_dataset():
    payloads, stats = public_payloads()
    for relative, content in payloads.items():
        path = SYNTHETIC_ROOT/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(content)
    return stats

def audit_public_dataset():
    """Só permite publicar o corpus exatamente regenerável pelo gerador revisado."""
    payloads, _ = public_payloads()
    for relative, expected in payloads.items():
        path = SYNTHETIC_ROOT/relative
        if not path.exists() or path.read_bytes()!=expected:
            raise ValueError(f'Corpus modificado ou não reprodutível: data/synthetic/{relative}')
    return len(payloads)
