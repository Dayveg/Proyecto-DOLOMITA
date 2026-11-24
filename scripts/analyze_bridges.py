import json
import re
from pathlib import Path

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

retrieval_records = json.load(open("retrieval_index.json", encoding="utf-8"))
doc_chunk_map = {}
for r in retrieval_records:
    doc_chunk_map.setdefault(r["doc_id"], []).append(r["text"])

PROTO_MAP = {f"MC26-SEC-{i:03d}": f"MC26-DOC-{24+i:03d}" for i in range(1, 21)}
ROLE_CHARTERS = {
    "chief of competition": "MC26-DOC-059",
    "venue manager": "MC26-DOC-060",
    "technical delegate": "MC26-DOC-061",
    "medical team lead": "MC26-DOC-062",
    "security lead": "MC26-DOC-063",
    "weather officer": "MC26-DOC-064",
    "logistics lead": "MC26-DOC-065",
}

VENUE_PREFIXES = {
    "milano": ("MC26-DOC-001", "MC26-DOC-002", "MC26-DOC-003", "MC26-DOC-004"),
    "cortina": ("MC26-DOC-005", "MC26-DOC-006", "MC26-DOC-007", "MC26-DOC-008"),
    "bormio": ("MC26-DOC-009", "MC26-DOC-010", "MC26-DOC-011", "MC26-DOC-012"),
    "livigno": ("MC26-DOC-013", "MC26-DOC-014", "MC26-DOC-015", "MC26-DOC-016"),
    "fiemme": ("MC26-DOC-017", "MC26-DOC-018", "MC26-DOC-019", "MC26-DOC-020"),
    "anterselva": ("MC26-DOC-021", "MC26-DOC-022", "MC26-DOC-023", "MC26-DOC-024"),
}

# Ver los 141 bridges que fallaron
failed_bridges = []
for x in train_jsonl:
    if x["type"] == "bridge":
        targets = set(x["support_doc_ids"])
        # Simular routing
        ql = x["question"].lower()
        matched = set()
        for p, did in PROTO_MAP.items():
            if p.lower() in ql: matched.add(did)
        for v in VENUE_PREFIXES:
            if v in ql:
                matched.add(VENUE_PREFIXES[v][1]) # ops
                matched.add(VENUE_PREFIXES[v][3]) # staff
                matched.add(VENUE_PREFIXES[v][2]) # sys
        if not targets.issubset(matched):
            failed_bridges.append((x["id"], x["question"], targets, x.get("bridge")))

print(f"Total bridges fallidos sin salto 2: {len(failed_bridges)}")
print("\nPrimeros 15 bridges fallidos:")
for f in failed_bridges[:15]:
    print(f"Q: {f[1]}")
    print(f"   Targets: {f[2]} | Bridge: {f[3]}")
