import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

bridges = [x for x in train_jsonl if x["type"] == "bridge"]
print(f"Total bridge en train: {len(bridges)}")

# Analizar la entidad puente
bridge_types = Counter()
for b in bridges:
    br = b.get("bridge", "")
    if re.match(r"^MC26-SEC-\d+", br):
        bridge_types["protocol_bridge"] += 1
    else:
        bridge_types[f"role_bridge: {br}"] += 1

print("\nDistribución de puentes:")
for k, v in bridge_types.most_common():
    print(f"  {k}: {v}")

print("\n--- MUESTRAS DE PUENTES DE PROTOCOLO ---")
for b in [x for x in bridges if re.match(r"^MC26-SEC-\d+", x.get("bridge", ""))][:10]:
    print(f"Q: {b['question']}")
    print(f"   Docs: {b['support_doc_ids']} | Bridge: {b.get('bridge')} | Ans: {b['answer']}")

print("\n--- MUESTRAS DE PUENTES DE ROL ---")
for b in [x for x in bridges if not re.match(r"^MC26-SEC-\d+", x.get("bridge", ""))][:10]:
    print(f"Q: {b['question']}")
    print(f"   Docs: {b['support_doc_ids']} | Bridge: {b.get('bridge')} | Ans: {b['answer']}")
