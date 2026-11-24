from pathlib import Path
import re

KB_DIR = Path("data/base_conocimiento")

# Analizar DOC 059 a 065 (Charters de los 6 roles)
charter_docs = {
    "chief of competition": "MC26-DOC-059",
    "venue manager": "MC26-DOC-060",
    "technical delegate": "MC26-DOC-061",
    "medical team lead": "MC26-DOC-062",
    "security lead": "MC26-DOC-063",
    "weather officer": "MC26-DOC-064",
}

print("=== REPORTE Y ESCALAMIENTO POR ROL (CHARTERS) ===")
for role, did in charter_docs.items():
    txt = (KB_DIR / did / "doc.md").read_text(encoding="utf-8")
    
    # Escalation deadline
    esc_m = re.search(r"within \*\*(\d+\s*minutes)\*\*", txt, re.IGNORECASE)
    esc_time = esc_m.group(1) if esc_m else "N/A"
    
    # Reporting line
    rep_m = re.search(r"reports directly to \*\*(.+?)\*\*", txt, re.IGNORECASE)
    reporting = rep_m.group(1) if rep_m else "N/A"
    
    print(f"Role: {role}")
    print(f"  Reporting line: '{reporting}'")
    print(f"  Escalation deadline: '{esc_time}'")
