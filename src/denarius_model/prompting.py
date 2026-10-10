import json

def schema(categories):
    if not categories or len(set(categories))!=len(categories): raise ValueError('Categorias inválidas')
    return {'type':'object','properties':{'category':{'type':'string','enum':categories}},
            'required':['category'],'additionalProperties':False}

def prompt(row,categories):
    schema(categories)
    # Plain completion format is identical in HF and Ollama raw mode.
    return ('Classifica o movimento. Devolve apenas um JSON com a chave category e uma categoria permitida.\n'
            'Categorias: '+json.dumps(categories,ensure_ascii=False)+'\n'
            'Movimento: '+json.dumps({k:row[k] for k in ('description','amount','direction')},ensure_ascii=False)+'\nResposta:')

def example(row,categories):
    if row['category'] not in categories: raise ValueError('Categoria desconhecida')
    return {'prompt':prompt(row,categories),'completion':json.dumps({'category':row['category']},ensure_ascii=False)}

def parse_output(text,categories):
    def pairs(items):
        if len(dict(items))!=len(items): raise ValueError('Chaves duplicadas')
        return dict(items)
    value=json.loads(text,object_pairs_hook=pairs)
    if not isinstance(value,dict) or set(value)!={'category'} or value['category'] not in categories:
        raise ValueError('Output fora do contrato')
    return value['category']
