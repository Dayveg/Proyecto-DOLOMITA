import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

bridges = [x for x in train_jsonl if x["type"] == "bridge"]
print(f"Total bridge: {len(bridges)}")

sample_bridges = bridges[:20]
for b in sample_bridges:
    print(f"Q: {b['question']}")
    print(f"   Doc1 -> Doc2: {b['support_doc_ids']} | Bridge Entity: '{b.get('bridge')}' | Ans: {b['answer']}")
