from pathlib import Path
import json
import subprocess
import pytest
from denarius_model import ROOT
from denarius_model.git_audit import audit_candidates
from denarius_model import synthetic

@pytest.mark.parametrize('name',['data/raw/movements.json','data/processed/categories.json','data/splits/train.json','data/synthetic/other.json','models/adapter.safetensors','outputs/experiment.json','reports/real.md','.env'])
def test_private_paths_are_rejected_even_if_force_added(name):
    with pytest.raises(ValueError): audit_candidates([name])

@pytest.mark.parametrize('name',['data/raw/movements.json','data/synthetic/other.json','models/final.gguf','outputs/experiment.json','.env'])
def test_gitignore_keeps_private_files_out(name):
    assert subprocess.run(['git','check-ignore','-q',name],cwd=ROOT).returncode==0

def test_public_synthetic_file_is_not_ignored():
    assert subprocess.run(['git','check-ignore','-q','data/synthetic/splits/train.json'],cwd=ROOT).returncode==1

def test_public_marker_does_not_let_modified_data_pass(tmp_path,monkeypatch):
    payloads,_=synthetic.public_payloads()
    monkeypatch.setattr(synthetic,'SYNTHETIC_ROOT',tmp_path)
    for relative,content in payloads.items():
        p=tmp_path/relative; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(content)
    synthetic.audit_public_dataset()
    p=tmp_path/'splits/train.json'
    rows=json.loads(p.read_text(encoding='utf-8'))
    rows[0]['description']='DADO INSERIDO MANUALMENTE'
    assert rows[0]['synthetic'] is True
    p.write_text(json.dumps(rows),encoding='utf-8')
    with pytest.raises(ValueError): synthetic.audit_public_dataset()

def test_saved_notebook_outputs_are_rejected(tmp_path):
    p=tmp_path/'analysis.ipynb'
    p.write_text(json.dumps({'cells':[{'outputs':[{'text':'sensitive'}],'execution_count':1}]}))
    with pytest.raises(ValueError): audit_candidates(['analysis.ipynb'],root=tmp_path)

def test_gitkeep_cannot_hide_nonempty_data(tmp_path):
    folder=tmp_path/'data'; folder.mkdir()
    (folder/'.gitkeep').write_text('NOT EMPTY')
    with pytest.raises(ValueError): audit_candidates(['data/.gitkeep'],root=tmp_path)
