from pathlib import Path
import re

KB_DIR = Path("data/base_conocimiento")

# Protocolos 001 a 020 (DOC 025 a 044)
print("=== PROTOCOLOS Y ACCIONES / AUTORIZACIONES ===")
for i in range(1, 21):
    sec = f"MC26-SEC-{i:03d}"
    did = f"MC26-DOC-{24+i:03d}"
    txt = (KB_DIR / did / "doc.md").read_text(encoding="utf-8")
    
    # First action (Paso 1)
    step1_m = re.search(r"\|\s*1\s*\|\s*([^|]+)\s*\|", txt)
    first_action = step1_m.group(1).strip() if step1_m else "N/A"
    
    # Resumption authorization
    auth_m = re.search(r"Resumption Authorization.*?(?:until|authorized by)\s+(?:the\s+)?\*\*?([a-zA-Z\s]+?)\*\*?(?:\s+has|\s+before|\.|\n)", txt, re.DOTALL | re.IGNORECASE)
    auth_role = auth_m.group(1).strip() if auth_m else "N/A"
    
    # Scope (venue_only vs games_wide)
    scope_m = re.search(r"\b(venue_only|games_wide)\b", txt, re.IGNORECASE)
    scope = scope_m.group(1).strip() if scope_m else "N/A"
    
    # Backup resource
    bk_m = re.search(r"Backup Resource.*?(?:is|designated backup.*?is)\s+(?:the\s+)?\*\*?([a-zA-Z\s\-]+?)\*\*?(?:\.|\n)", txt, re.DOTALL | re.IGNORECASE)
    backup = bk_m.group(1).strip() if bk_m else "N/A"
    
    print(f"{sec} ({did}):")
    print(f"  Step 1: '{first_action}'")
    print(f"  Resumption auth: '{auth_role}'")
    print(f"  Scope: '{scope}'")
    print(f"  Backup: '{backup}'")
