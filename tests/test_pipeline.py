import json
import pytest
from denarius_model.preprocessing import normalize,redact
from denarius_model.prompting import prompt,example,parse_output,schema
from denarius_model.synthetic import generate_rows,validate_splits,public_payloads,GENERATOR_VERSION

def row(i=0):
    return {'description':'LOJA '+chr(65+i)*8,'debit':10,'credit':0,'category':'Grupo / Compras','entry_group':str(i)}

def test_normalization_whitelist():
    r=row(); r.update(iban='PT50000000000000000000000',name='Pessoa')
    out=normalize(r)
    assert out['amount']=='10' and out['direction']=='debito_contabilistico'
    assert 'iban' not in out and 'name' not in out

def test_redaction():
    assert 'PT50' not in redact('IBAN PT50000000000000000000000')
    assert '@' not in redact('a@b.pt 912345678')

@pytest.mark.parametrize('d,c',[(0,0),(1,1),(-1,0),('NaN',0),('Infinity',0)])
def test_invalid_amount(d,c):
    r=row(); r.update(debit=d,credit=c)
    with pytest.raises(ValueError): normalize(r)

@pytest.mark.parametrize('text',['```json\n{"category":"A"}\n```','{"category":"X"}','{"category":"A","confidence":1}','{"category":["A"]}','{"category":"A"} extra','{"category":"A","category":"A"}','[]'])
def test_invalid_output(text):
    with pytest.raises((ValueError,TypeError)): parse_output(text,['A'])

def test_example():
    r=normalize(row()); cats=[r['category']]
    assert parse_output(example(r,cats)['completion'],cats)==r['category']
    assert 'entry_group' not in prompt(r,cats)
    assert schema(cats)['additionalProperties'] is False
    with pytest.raises(ValueError): example(r,['Other'])

def small_corpus(seed=42):
    labels=['Despesas Correntes / Combustível','Rendimentos Correntes / Salário']
    cfg={'seed':seed,'generator_version':GENERATOR_VERSION,'per_category':{'train':40,'validation':8,'test':8,'hard_test':8}}
    return generate_rows(labels,cfg)

def test_generation_deterministic_and_seed_sensitive():
    assert small_corpus()==small_corpus()
    assert small_corpus()!=small_corpus(43)

def test_scenario_groups_never_cross_splits():
    rows=small_corpus()
    groups={}
    for r in rows: groups.setdefault(r['entry_group'],set()).add(r['split'])
    assert all(len(s)==1 for s in groups.values())
    splits={s:[{**normalize(r),'counterparty':r['counterparty'],'template_family':r['template_family']} for r in rows if r['split']==s] for s in ('train','validation','test','hard_test')}
    validate_splits(splits,sorted({r['category'] for r in rows}))

def test_balanced_labels_and_valid_outputs():
    from collections import Counter
    rows=small_corpus()
    for split in ('train','validation','test','hard_test'):
        counts=Counter(r['category'] for r in rows if r['split']==split)
        assert len(set(counts.values()))==1
    cats=sorted({r['category'] for r in rows})
    for r in rows:
        assert r['synthetic'] is True and r['source_version']==GENERATOR_VERSION
        assert (r['debit']==0)!=(r['credit']==0)
        assert parse_output(example(normalize(r),cats)['completion'],cats)==r['category']

def test_raw_descriptions_have_case_spacing_and_accent_variants():
    rows=small_corpus()
    text=[r['description'] for r in rows]
    assert any('  ' in t for t in text)
    assert any(t!=t.upper() for t in text)
    assert any(t==t.upper() for t in text)
    assert any('GASÓLEO' in t.upper() for t in text)
    assert any('GASOLEO' in t.upper() for t in text)

def test_hard_counterparties_and_templates_are_unseen():
    rows=small_corpus()
    hard=[r for r in rows if r['split']=='hard_test']
    normal=[r for r in rows if r['split']!='hard_test']
    for field in ('counterparty','template_family'):
        assert not {r[field] for r in hard}&{r[field] for r in normal}

def test_too_many_examples_fail_instead_of_looping():
    cfg={'seed':42,'per_category':dict(train=10000,validation=8,test=8,hard_test=8)}
    with pytest.raises(ValueError): generate_rows(['Despesas Correntes / Combustível'],cfg)

def test_corpus_complete_and_matches_taxonomy():
    from denarius_model.synthetic import taxonomy
    payloads,stats=public_payloads()
    labels=taxonomy()
    assert len(labels)==70
    assert stats['source']=='synthetic'
    for split in ('train','validation','test','hard_test'):
        rows=json.loads(payloads[f'splits/{split}.json'])
        assert set(r['category'] for r in rows)==set(labels)
        assert all(r['synthetic'] is True for r in rows)
