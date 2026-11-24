import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from scripts.normalizer import normalize_answer

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

# 1. Comparaciones de umbrales entre 2 sedes (100% exacto)
from scripts.test_comparison_solver import solve_comparison

# 2. Las 4 dimensiones canónicas de cada uno de los 20 protocolos
PROTO_CANONICAL = {
    "MC26-SEC-001": {
        "scope": "venue only",
        "action": "suspend outdoor activity",
        "backup": "heated shelter",
        "resumption": "chief of competition"
    },
    "MC26-SEC-002": {
        "scope": "games wide",
        "action": "hold the competition",
        "backup": "wind sensor array",
        "resumption": "venue manager"
    },
    "MC26-SEC-003": {
        "scope": "venue only",
        "action": "delay the start",
        "backup": "fog lighting",
        "resumption": "technical delegate"
    },
    "MC26-SEC-004": {
        "scope": "games wide",
        "action": "activate snowmaking",
        "backup": "snow additive",
        "resumption": "medical team lead"
    },
    "MC26-SEC-005": {
        "scope": "venue only",
        "action": "switch to backup system",
        "backup": "spare parts kit",
        "resumption": "security lead"
    },
    "MC26-SEC-006": {
        "scope": "games wide",
        "action": "start the diesel generator",
        "backup": "diesel generator",
        "resumption": "weather officer"
    },
    "MC26-SEC-007": {
        "scope": "venue only",
        "action": "dispatch the medical team",
        "backup": "ambulance",
        "resumption": "logistics lead"
    },
    "MC26-SEC-008": {
        "scope": "games wide",
        "action": "lock down the venue",
        "backup": "radio network",
        "resumption": "chief of competition"
    },
    "MC26-SEC-009": {
        "scope": "venue only",
        "action": "muster point",
        "backup": "muster point",
        "resumption": "venue manager"
    },
    "MC26-SEC-010": {
        "scope": "games wide",
        "action": "close the slope",
        "backup": "avalanche transceiver",
        "resumption": "technical delegate"
    },
    "MC26-SEC-011": {
        "scope": "venue only",
        "action": "clear the ice surface",
        "backup": "ice resurfacer",
        "resumption": "medical team lead"
    },
    "MC26-SEC-012": {
        "scope": "games wide",
        "action": "switch to the radio network",
        "backup": "radio network",
        "resumption": "security lead"
    },
    "MC26-SEC-013": {
        "scope": "venue only",
        "action": "open additional exits",
        "backup": "barrier system",
        "resumption": "weather officer"
    },
    "MC26-SEC-014": {
        "scope": "games wide",
        "action": "activate the fire suppression system",
        "backup": "fire suppression system",
        "resumption": "logistics lead"
    },
    "MC26-SEC-015": {
        "scope": "venue only",
        "action": "reroute the shuttle fleet",
        "backup": "shuttle fleet",
        "resumption": "chief of competition"
    },
    "MC26-SEC-016": {
        "scope": "games wide",
        "action": "escort the athlete to the control station",
        "backup": "chain-of-custody kit",
        "resumption": "venue manager"
    },
    "MC26-SEC-017": {
        "scope": "venue only",
        "action": "raise the security level",
        "backup": "barrier system",
        "resumption": "technical delegate"
    },
    "MC26-SEC-018": {
        "scope": "games wide",
        "action": "alert the venue manager",
        "backup": "tracking beacon",
        "resumption": "medical team lead"
    },
    "MC26-SEC-019": {
        "scope": "venue only",
        "action": "activate the counter-drone system",
        "backup": "counter-drone system",
        "resumption": "security lead"
    },
    "MC26-SEC-020": {
        "scope": "games wide",
        "action": "move personnel to shelter",
        "backup": "heated shelter",
        "resumption": "weather officer"
    }
}

# 3. Charters de roles
CHARTER_KB = {
    "chief of competition": {"reporting": "competition management", "deadline": "10 minutes"},
    "venue manager": {"reporting": "games operations center", "deadline": "15 minutes"},
    "technical delegate": {"reporting": "international federation", "deadline": "20 minutes"},
    "medical team lead": {"reporting": "medical services", "deadline": "5 minutes"},
    "security lead": {"reporting": "security operations", "deadline": "8 minutes"},
    "weather officer": {"reporting": "weather operations", "deadline": "12 minutes"},
    "logistics lead": {"reporting": "games operations center", "deadline": "24 hours"},
}

# Mapeos de área de sede al rol
STAFF_AREAS = [
    ("timing calibration", "chief of competition"),
    ("timing", "chief of competition"),
    ("snow grooming", "venue manager"),
    ("slope grooming", "venue manager"),
    ("grooming", "venue manager"),
    ("field of play safety", "technical delegate"),
    ("field of play", "technical delegate"),
    ("first aid", "medical team lead"),
    ("access control", "security lead"),
    ("access", "security lead"),
    ("weather forecasting", "weather officer"),
    ("weather", "weather officer"),
    ("volunteer", "logistics lead"),
]

# Mapeo de condiciones climáticas de sede al protocolo
VENUE_COND_PROTO = [
    ("extreme cold", "MC26-SEC-001"),
    ("cold", "MC26-SEC-001"),
    ("high wind", "MC26-SEC-002"),
    ("wind", "MC26-SEC-002"),
    ("low visibility", "MC26-SEC-003"),
    ("visibility", "MC26-SEC-003"),
    ("snow shortage", "MC26-SEC-004"),
    ("avalanche", "MC26-SEC-010"),
    ("ice failure", "MC26-SEC-011"),
    ("ice", "MC26-SEC-011"),
    ("crowd", "MC26-SEC-013"),
    ("fire", "MC26-SEC-014"),
    ("transport", "MC26-SEC-015"),
    ("doping", "MC26-SEC-016"),
    ("severe storm", "MC26-SEC-020"),
    ("storm", "MC26-SEC-020"),
]

# 4. Umbrales numéricos de las sedes
OPS_THRESHOLDS = {
    "cortina": {"snow": "45 cm", "wind": "100 km/h", "visibility": "200 m", "temp": "-22 c"},
    "bormio": {"snow": "40 cm", "wind": "110 km/h", "visibility": "250 m", "temp": "-25 c"},
    "livigno": {"snow": "50 cm", "wind": "80 km/h", "visibility": "150 m", "temp": "-28 c"},
    "fiemme": {"snow": "35 cm", "wind": "90 km/h", "visibility": "300 m", "temp": "-20 c"},
    "anterselva": {"snow": "30 cm", "wind": "70 km/h", "visibility": "180 m", "temp": "-24 c"},
    "milano": {"temp": "-5 c", "humidity": "60 %", "resurfacing": "30 minutes"}
}

# 5. Sistemas de backup de las sedes
VENUE_BACKUPS = {
    "milano": {"timing": "uninterruptible power supply", "lighting": "diesel generator", "snowmaking": "water reservoir", "security": "radio network", "transport": "shuttle fleet"},
    "cortina": {"timing": "uninterruptible power supply", "lighting": "diesel generator", "snowmaking": "water reservoir", "security": "radio network", "transport": "shuttle fleet"},
    "bormio": {"timing": "uninterruptible power supply", "lighting": "diesel generator", "snowmaking": "water reservoir", "security": "radio network", "transport": "shuttle fleet"},
    "livigno": {"timing": "uninterruptible power supply", "lighting": "diesel generator", "snowmaking": "water reservoir", "security": "radio network", "transport": "shuttle fleet"},
    "fiemme": {"timing": "uninterruptible power supply", "lighting": "diesel generator", "snowmaking": "water reservoir", "security": "radio network", "transport": "shuttle fleet"},
    "anterselva": {"timing": "uninterruptible power supply", "lighting": "diesel generator", "snowmaking": "water reservoir", "security": "radio network", "transport": "shuttle fleet"}
}

VENUES = ["milano", "cortina", "bormio", "livigno", "fiemme", "anterselva"]

def solve_deterministic(q: str):
    ql = q.lower()
    
    # A. Comparación entre sedes
    comp = solve_comparison(q)
    if comp:
        return comp

    # B. Pregunta sobre protocolo específico (MC26-SEC-XXX)
    proto_m = re.search(r"MC26-SEC-\d+", ql)
    if proto_m:
        proto = proto_m.group(0).upper()
        if proto in PROTO_CANONICAL:
            info = PROTO_CANONICAL[proto]
            if any(w in ql for w in ["scope", "apply", "cover", "encompass", "operations", "functions"]):
                return info["scope"]
            if any(w in ql for w in ["action", "step", "first", "initial", "opening"]):
                return info["action"]
            if any(w in ql for w in ["backup", "redundanc", "equipment"]):
                return info["backup"]
            if any(w in ql for w in ["resumption", "resume", "restart", "sign off", "green light", "call", "authoriz"]):
                return info["resumption"]

    # C. Pregunta de reporte o escalamiento de rol (directo o via sede)
    is_reporting = any(w in ql for w in ["report", "boss", "answers to", "reporting line"])
    is_deadline = any(w in ql for w in ["escalat", "deadline", "window"])
    
    target_role = None
    for r in CHARTER_KB:
        if r in ql:
            target_role = r
            break
    if not target_role:
        for area, r in STAFF_AREAS:
            if area in ql:
                target_role = r
                break
                
    if target_role:
        if is_reporting:
            return CHARTER_KB[target_role]["reporting"]
        if is_deadline:
            return CHARTER_KB[target_role]["deadline"]

    # D. Pregunta sobre una sede específica
    venues_in_q = [v for v in VENUES if v in ql]
    if len(venues_in_q) == 1:
        v = venues_in_q[0]
        
        # D.1. Protocolo puente en sede ante contingencia
        is_action = any(w in ql for w in ["action", "step", "first", "initial", "what do we do"])
        is_resume = any(w in ql for w in ["resume", "resumption", "restart", "sign off", "green light", "authority", "who can give"])
        if is_action or is_resume:
            for cond, proto in VENUE_COND_PROTO:
                if cond in ql and proto in PROTO_CANONICAL:
                    if is_action: return PROTO_CANONICAL[proto]["action"]
                    if is_resume: return PROTO_CANONICAL[proto]["resumption"]

        # D.2. Responsabilidad de personal en la sede
        if any(w in ql for w in ["who", "responsible", "lead", "boss", "manage", "oversee", "owns"]):
            for area, r in STAFF_AREAS:
                if area in ql:
                    return r
                    
        # D.3. Umbrales de la sede
        if v in OPS_THRESHOLDS:
            th = OPS_THRESHOLDS[v]
            if any(w in ql for w in ["snow", "depth"]) and "snow" in th: return th["snow"]
            if any(w in ql for w in ["wind", "speed"]) and "wind" in th: return th["wind"]
            if any(w in ql for w in ["visib", "sight"]) and "visibility" in th: return th["visibility"]
            if any(w in ql for w in ["temp", "cold"]) and "temp" in th: return th["temp"]
            if "humidity" in ql and "humidity" in th: return th["humidity"]
            if "resurfac" in ql and "resurfacing" in th: return th["resurfacing"]

        # D.4. Sistemas de backup en la sede
        if any(w in ql for w in ["backup", "power", "lighting", "ups", "generator", "reservoir", "secondary", "solution"]):
            if any(w in ql for w in ["time", "timing"]): return VENUE_BACKUPS[v]["timing"]
            if any(w in ql for w in ["light", "lighting"]): return VENUE_BACKUPS[v]["lighting"]
            if any(w in ql for w in ["snow", "snowmaking"]): return VENUE_BACKUPS[v]["snowmaking"]
            if any(w in ql for w in ["sec", "security"]): return VENUE_BACKUPS[v]["security"]
            if any(w in ql for w in ["trans", "transport", "vehicle"]): return VENUE_BACKUPS[v]["transport"]

    # E. Preguntas sobre pista de Milano (rink)
    if "rink" in ql:
        if "humidity" in ql: return "60 %"
        if "temp" in ql: return "-5 c"
        if "resurfac" in ql: return "30 minutes"

    return None

# Evaluar sobre TODO el train (2,940 preguntas)
hits = 0
unsolved = 0
for x in train_jsonl:
    p = solve_deterministic(x["question"])
    g = normalize_answer(x["answer"])
    if p:
        if normalize_answer(p) == g:
            hits += 1
        else:
            # error
            pass
    else:
        unsolved += 1

print(f"Total train: {len(train_jsonl)}")
print(f"Resueltas por base de conocimiento determinista: {len(train_jsonl) - unsolved}/{len(train_jsonl)}")
print(f"Exact Match en train: {hits}/{len(train_jsonl)} ({hits/len(train_jsonl)*100:.2f}%)")
print(f"Sin resolver directo (para LLM/RAG): {unsolved}")
