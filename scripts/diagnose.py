import json
import re
import pandas as pd
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

# Analizar tipos de preguntas en train
by_type = Counter(x["type"] for x in train_jsonl)
print("Tipos de preguntas en train:", by_type)

# Muestras de bridge
print("\n--- EJEMPLOS BRIDGE ---")
for x in [x for x in train_jsonl if x["type"] == "bridge"][:5]:
    print(f"Q: {x['question']}")
    print(f"  Docs: {x['support_doc_ids']} | Bridge: {x.get('bridge')} | Ans: {x['answer']}")

# Muestras de comparison
print("\n--- EJEMPLOS COMPARISON ---")
for x in [x for x in train_jsonl if x["type"] == "comparison"][:5]:
    print(f"Q: {x['question']}")
    print(f"  Docs: {x['support_doc_ids']} | Ans: {x['answer']}")

# Muestras de single
print("\n--- EJEMPLOS SINGLE ---")
for x in [x for x in train_jsonl if x["type"] == "single"][:5]:
    print(f"Q: {x['question']}")
    print(f"  Docs: {x['support_doc_ids']} | Ans: {x['answer']}")
