import json
from pathlib import Path
import jupytext

nb_path = Path("pipeline.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

cell_1_src = nb["cells"][1]["source"]
insert_idx = -1
for i, line in enumerate(cell_1_src):
    if "SEED = 42" in line:
        insert_idx = i
        break

if insert_idx != -1:
    cell_1_src.insert(insert_idx, "os.environ['HF_HUB_OFFLINE'] = '1'\n")
    cell_1_src.insert(insert_idx + 1, "os.environ['TRANSFORMERS_OFFLINE'] = '1'\n")

nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# Convertir a script
py_code = jupytext.writes(jupytext.read(str(nb_path)), fmt="py:percent")
Path("pipeline.py").write_text(py_code, encoding="utf-8")
print("Modo offline aplicado con éxito a pipeline.ipynb y pipeline.py")
