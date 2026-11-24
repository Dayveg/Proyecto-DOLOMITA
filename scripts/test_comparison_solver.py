import json
import re
from pathlib import Path

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

comparisons = [x for x in train_jsonl if x["type"] == "comparison"]
print(f"Total comparaciones en train: {len(comparisons)}")

# Diccionario exacto de umbrales extraído de los documentos técnicos oficiales (DOC 006, 010, 014, 018, 022)
THRESHOLDS = {
    # Snow depth (Course readiness minimum cm)
    "snow": {
        "livigno": 50,
        "cortina": 45,
        "bormio": 40,
        "fiemme": 35,
        "anterselva": 30
    },
    # Wind speed (Max operational limit km/h)
    "wind": {
        "bormio": 110,
        "cortina": 100,
        "fiemme": 90,
        "livigno": 80,
        "anterselva": 70
    },
    # Visibility (Minimum start-gate visibility m)
    "visibility": {
        "fiemme": 300,
        "bormio": 250,
        "cortina": 200,
        "anterselva": 180,
        "livigno": 150
    },
    # Temperature (Suspension threshold in Celsius)
    "temp": {
        "anterselva": -24,
        "bormio": -25,
        "livigno": -28,
        "cortina": -22,
        "fiemme": -20
    }
}

VENUES = ["milano", "cortina", "bormio", "livigno", "fiemme", "anterselva"]

def solve_comparison(q):
    ql = q.lower()
    found_venues = [v for v in VENUES if v in ql]
    if len(found_venues) != 2:
        return None
    
    v1, v2 = found_venues[0], found_venues[1]
    
    # Determinar qué métrica se está comparando
    metric = None
    if any(w in ql for w in ["snow", "depth"]):
        metric = "snow"
    elif any(w in ql for w in ["wind", "speed"]):
        metric = "wind"
    elif any(w in ql for w in ["visib", "sight"]):
        metric = "visibility"
    elif any(w in ql for w in ["temp", "cold"]):
        metric = "temp"
        
    if not metric or metric not in THRESHOLDS:
        return None
        
    val1 = THRESHOLDS[metric].get(v1)
    val2 = THRESHOLDS[metric].get(v2)
    if val1 is None or val2 is None:
        return None
        
    # Determinar dirección de comparación (mayor vs menor)
    # Por defecto en las preguntas comparativas se pregunta "higher", "greater", "beat", "more", "better"
    is_higher = any(w in ql for w in ["higher", "greater", "more", "better", "beat", "above", "tolerat"])
    is_lower = any(w in ql for w in ["lower", "less", "smaller", "below"])
    
    if is_lower:
        return v1 if val1 < val2 else v2
    else:
        return v1 if val1 > val2 else v2

# Evaluar en las 291 comparaciones de train
correct = 0
for c in comparisons:
    pred = solve_comparison(c["question"])
    gold = c["answer"].strip().lower()
    if pred == gold:
        correct += 1
    else:
        print(f"FAIL: Q='{c['question']}'")
        print(f"      PRED={pred} vs GOLD={gold}")

print(f"\nExactitud en las comparaciones de train: {correct}/{len(comparisons)} ({correct/len(comparisons)*100:.2f}%)")
