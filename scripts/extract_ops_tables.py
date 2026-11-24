import re
from pathlib import Path

# Cargar las 6 páginas de Operations & Thresholds de las 6 sedes
# MC26-DOC-002: Milano, 006: Cortina, 010: Bormio, 014: Livigno, 018: Fiemme, 022: Anterselva
KB_DIR = Path("data/base_conocimiento")
ops_docs = {
    "milano": KB_DIR / "MC26-DOC-002" / "doc.md",
    "cortina": KB_DIR / "MC26-DOC-006" / "doc.md",
    "bormio": KB_DIR / "MC26-DOC-010" / "doc.md",
    "livigno": KB_DIR / "MC26-DOC-014" / "doc.md",
    "fiemme": KB_DIR / "MC26-DOC-018" / "doc.md",
    "anterselva": KB_DIR / "MC26-DOC-022" / "doc.md",
}

ops_values = {}
for venue, p in ops_docs.items():
    txt = p.read_text(encoding="utf-8")
    # Snow depth (Course readiness): minimum of **45 cm**
    snow_m = re.search(r"snow depth.*?minimum of \*\*(\d+)\s*cm\*\*", txt, re.IGNORECASE | re.DOTALL)
    snow = int(snow_m.group(1)) if snow_m else None
    
    # Wind speed: **100 km/h** o en la tabla
    wind_m = re.search(r"Wind speed.*?(\d+)\s*km/h", txt, re.IGNORECASE)
    wind = int(wind_m.group(1)) if wind_m else None
    
    # Visibility: **200 m** o en la tabla
    vis_m = re.search(r"visibility.*?(\d+)\s*m\b", txt, re.IGNORECASE)
    vis = int(vis_m.group(1)) if vis_m else None
    
    # Temperature: −22 °C
    temp_m = re.search(r"temperature.*?(-?\d+)\s*°\s*C", txt, re.IGNORECASE)
    temp = int(temp_m.group(1)) if temp_m else None
    
    ops_values[venue] = {
        "snow": snow,
        "wind": wind,
        "visibility": vis,
        "temp": temp
    }

print("Valores de umbrales extraídos por sede:")
for k, v in ops_values.items():
    print(f"  {k}: {v}")
