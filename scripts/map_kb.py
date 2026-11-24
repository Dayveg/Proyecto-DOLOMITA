import re
from pathlib import Path
import json

DATA_DIR = Path("data")
KB_DIR = DATA_DIR / "base_conocimiento"

# Documentos 001 a 024 son de las 6 sedes (4 por sede: Overview, Ops/Thresholds, Systems, Staffing)
# 025 a 044 son los 20 protocolos MC26-SEC-001 a 020
# 059 a 065 son los 7 charters de roles

venues = ["milano", "cortina", "bormio", "livigno", "fiemme", "anterselva"]

doc_map = {}
for p in sorted(KB_DIR.glob("*/doc.md")):
    did = p.parent.name
    txt = p.read_text(encoding="utf-8")
    doc_map[did] = txt

print(f"Total docs cargados: {len(doc_map)}")

# Protocolos en documentos
proto_to_doc = {}
for i in range(1, 21):
    sec_id = f"MC26-SEC-{i:03d}"
    doc_id = f"MC26-DOC-{24+i:03d}"
    proto_to_doc[sec_id] = doc_id

print("Mapeo protocolo -> documento:")
print(list(proto_to_doc.items())[:5])

# Roles charters
role_charters = {
    "chief of competition": "MC26-DOC-059",
    "venue manager": "MC26-DOC-060",
    "technical delegate": "MC26-DOC-061",
    "medical team lead": "MC26-DOC-062",
    "security lead": "MC26-DOC-063",
    "weather officer": "MC26-DOC-064",
    "logistics lead": "MC26-DOC-065",
}
print("Mapeo rol charter -> documento:")
print(role_charters)
