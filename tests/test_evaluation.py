import json
import pytest
from denarius_model import evaluate as evaluation

def test_final_guard_does_not_read_test_records(monkeypatch):
    def forbidden_read(path): raise AssertionError('Teste foi lido antes da autorização de fase')
    monkeypatch.setattr(evaluation,'read_json',forbidden_read)
    with pytest.raises(ValueError): evaluation.evaluate(None,'test','base','synthetic-guard')
    with pytest.raises(ValueError): evaluation.evaluate(None,'hard_test','base','synthetic-guard')

def test_final_cannot_be_sampled():
    with pytest.raises(ValueError): evaluation.evaluate(None,'test','base','synthetic-guard',final=True,per_category=2)

def test_exploratory_sample_and_invalid_output_counting(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    def valid(row,categories): return json.dumps({'category':row['category']}),.01,{}
    def invalid(row,categories): return json.dumps({'category':row['category'],'confidence':.9}),.01,{}
    a=evaluation.evaluate(valid,'validation','valid','unit-test',per_category=1)
    b=evaluation.evaluate(invalid,'validation','invalid','unit-test',per_category=1)
    assert a['sample_size']==70 and a['accuracy']==1
    assert b['invalid_outputs']==70 and b['accuracy']==0
    assert a['evaluation_rows_hash']==b['evaluation_rows_hash']
    assert sum(sum(r) for r in b['confusion_matrix'])==70
