import json
from pathlib import Path

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

# Filtrar preguntas de train que consultan por protocolos MC26-SEC
sec_q = [x for x in train_jsonl if any(f"MC26-SEC-{i:03d}" in x["question"] for i in range(1, 21))]
print(f"Total preguntas de protocolos en train: {len(sec_q)}")

# Agrupar por protocolo y tipo de pregunta
by_proto = {}
for q in sec_q:
    ans = q["answer"]
    proto = [f"MC26-SEC-{i:03d}" for i in range(1, 21) if f"MC26-SEC-{i:03d}" in q["question"]][0]
    by_proto.setdefault(proto, set()).add(ans)

for p, answers in sorted(by_proto.items()):
    print(f"{p}: {answers}")
