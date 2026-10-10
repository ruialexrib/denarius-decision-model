import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from denarius_model.cli import main
sys.argv.insert(1,'generate')
main()
