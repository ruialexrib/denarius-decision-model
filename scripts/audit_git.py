import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import subprocess
from denarius_model import ROOT
from denarius_model.git_audit import audit_candidates

files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
try:
    count=audit_candidates(list(filter(None,files)))
except ValueError as exc:
    raise SystemExit(str(exc))
print(f'Auditoria: {count} ficheiros; corpus regenerável, sem paths privados ou outputs. Rever também o código antes do commit.')
