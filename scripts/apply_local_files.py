import json
from pathlib import Path
import jupytext

nb_path = Path("pipeline.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            line = line.replace(".from_pretrained(ENCODER_MODEL_ID)", ".from_pretrained(ENCODER_MODEL_ID, local_files_only=True)")
            line = line.replace(".from_pretrained(DECODER_MODEL_ID,", ".from_pretrained(DECODER_MODEL_ID, local_files_only=True,")
            new_source.append(line)
        cell["source"] = new_source

nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# Generar script sincronizado
py_code = jupytext.writes(jupytext.read(str(nb_path)), fmt="py:percent")
Path("pipeline.py").write_text(py_code, encoding="utf-8")
print("local_files_only=True aplicado en pipeline.ipynb y pipeline.py")
