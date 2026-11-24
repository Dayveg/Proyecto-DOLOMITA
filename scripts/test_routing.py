import json
import re
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

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

OP_AREAS = [
    ("timing calibration", "chief of competition"),
    ("snow grooming", "venue manager"),
    ("field of play safety", "technical delegate"),
    ("first aid", "medical team lead"),
    ("access control", "security lead"),
    ("weather forecasting", "weather officer"),
]

WEATHER_PROTO = [
    ("extreme cold", "MC26-SEC-001"),
    ("cold", "MC26-SEC-001"),
    ("high wind", "MC26-SEC-002"),
    ("wind", "MC26-SEC-002"),
    ("low visibility", "MC26-SEC-003"),
    ("visibility", "MC26-SEC-003"),
    ("snow shortage", "MC26-SEC-004"),
    ("ice failure", "MC26-SEC-011"),
    ("ice", "MC26-SEC-011"),
    ("storm", "MC26-SEC-020"),
    ("severe storm", "MC26-SEC-020"),
]

def perfect_routing(q):
    ql = q.lower()
    matched_docs = set()
    
    # 1. Si menciona directamente MC26-SEC-XXX
    for p, did in PROTO_MAP.items():
        if p.lower() in ql:
            matched_docs.add(did)
            
    # 2. Sedes mencionadas
    venues_found = [v for v in VENUE_PREFIXES if v in ql]
    
    # Si es comparación entre 2 sedes
    if len(venues_found) >= 2:
        for v in venues_found:
            matched_docs.add(VENUE_PREFIXES[v][1])
        return matched_docs
        
    if len(venues_found) == 1:
        v = venues_found[0]
        is_staffing = any(area[0] in ql for area in OP_AREAS) or any(w in ql for w in ["who", "responsible", "lead", "boss", "report", "escalat", "deadline", "manage", "oversee", "answers to"])
        is_ops = any(wp[0] in ql for wp in WEATHER_PROTO) or any(w in ql for w in ["wind", "snow", "temp", "visibility", "cutoff", "threshold", "minimum", "maximum", "first action", "first step", "resumption", "restart", "hold"])
        is_sys = any(w in ql for w in ["backup", "power", "lighting", "generator", "ups", "reservoir", "system", "infrastructure"])
        
        if is_staffing:
            matched_docs.add(VENUE_PREFIXES[v][3])
        if is_ops:
            matched_docs.add(VENUE_PREFIXES[v][1])
        if is_sys:
            matched_docs.add(VENUE_PREFIXES[v][2])
            
        for area, role in OP_AREAS:
            if area in ql:
                if any(w in ql for w in ["report", "boss", "escalat", "deadline", "answer to"]):
                    matched_docs.add(ROLE_CHARTERS[role])
                    
        for cond, proto in WEATHER_PROTO:
            if cond in ql:
                if any(w in ql for w in ["action", "step", "first", "resume", "resumption", "restart", "sign off", "green light", "hold"]):
                    matched_docs.add(PROTO_MAP[proto])
                    
    for role, charter_doc in ROLE_CHARTERS.items():
        if role in ql:
            matched_docs.add(charter_doc)
            
    return matched_docs

perfect_hits = 0
by_type_hits = Counter()

for x in train_jsonl:
    targets = set(x["support_doc_ids"])
    pred_docs = perfect_routing(x["question"])
    if targets.issubset(pred_docs):
        perfect_hits += 1
        by_type_hits[x["type"]] += 1

print(f"Total evaluados: {len(train_jsonl)}")
print(f"Cobertura Routing de Dominio: {perfect_hits}/{len(train_jsonl)} ({perfect_hits/len(train_jsonl)*100:.2f}%)")
print("Hits por tipo:", by_type_hits)
