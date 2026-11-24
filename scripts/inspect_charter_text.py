from pathlib import Path
import re

KB_DIR = Path("data/base_conocimiento")
charter_docs = {
    "chief of competition": "MC26-DOC-059",
    "venue manager": "MC26-DOC-060",
    "technical delegate": "MC26-DOC-061",
    "medical team lead": "MC26-DOC-062",
    "security lead": "MC26-DOC-063",
    "weather officer": "MC26-DOC-064",
}

for role, did in charter_docs.items():
    txt = (KB_DIR / did / "doc.md").read_text(encoding="utf-8")
    print(f"\n--- {role} ({did}) ---")
    for sec in txt.split("## "):
        if any(w in sec.lower() for w in ["escalat", "report"]):
            print("## " + sec[:400] + "...")
