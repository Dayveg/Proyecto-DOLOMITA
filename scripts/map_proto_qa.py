import json
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

proto_qa = defaultdict(lambda: defaultdict(list))
for x in train_jsonl:
    for did in x["support_doc_ids"]:
        if 25 <= int(did.split("-")[-1]) <= 44:
            proto_qa[did][x["answer"]].append(x["question"])

print(f"Total protocolos con preguntas en train: {len(proto_qa)}")
for did in sorted(proto_qa):
    sec = f"MC26-SEC-{int(did.split('-')[-1])-24:03d}"
    print(f"\n{sec} ({did}):")
    for ans, qs in proto_qa[did].items():
        print(f"   Ans: '{ans}' ({len(qs)} preguntas) -> Ej: '{qs[0]}'")
