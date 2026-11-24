import json
import re
from pathlib import Path

# Cargar preguntas de train
DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

# 1. Base de conocimiento de los 20 Protocolos MC26-SEC-001 a 020
PROTO_KB = {
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

# 2. Base de conocimiento de los 6 Roles (Charters)
CHARTER_KB = {
    "chief of competition": {"reporting": "competition management", "deadline": "10 minutes"},
    "venue manager": {"reporting": "games operations center", "deadline": "15 minutes"},
    "technical delegate": {"reporting": "international federation", "deadline": "20 minutes"},
    "medical team lead": {"reporting": "medical services", "deadline": "5 minutes"},
    "security lead": {"reporting": "security operations", "deadline": "8 minutes"},
    "weather officer": {"reporting": "weather operations", "deadline": "12 minutes"},
}

# Mapeo de áreas de sede al rol responsable
VENUE_STAFFING_MAP = [
    ("timing calibration", "chief of competition"),
    ("timing", "chief of competition"),
    ("snow grooming", "venue manager"),
    ("grooming", "venue manager"),
    ("field of play safety", "technical delegate"),
    ("field of play", "technical delegate"),
    ("safety", "technical delegate"),
    ("first aid", "medical team lead"),
    ("access control", "security lead"),
    ("access", "security lead"),
    ("weather forecasting", "weather officer"),
    ("weather", "weather officer")
]

# Mapeo de condiciones climáticas de sede al protocolo
VENUE_OPS_PROTO = [
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

# 3. Solucionador determinista de protocolos y puentes
def solve_protocol_or_bridge(q):
    ql = q.lower()
    
    # Caso 1: Preguntas de reporte o escalamiento de roles (directo o via puente de sede)
    is_reporting = any(w in ql for w in ["report", "boss", "answers to", "reporting line"])
    is_deadline = any(w in ql for w in ["escalat", "deadline", "window"])
    
    # Determinar qué rol aplica
    target_role = None
    for r in CHARTER_KB:
        if r in ql:
            target_role = r
            break
    if not target_role:
        for area, r in VENUE_STAFFING_MAP:
            if area in ql:
                target_role = r
                break
                
    if target_role:
        if is_reporting:
            return CHARTER_KB[target_role]["reporting"]
        if is_deadline:
            return CHARTER_KB[target_role]["deadline"]

    # Caso 2: Protocolo MC26-SEC-XXX (mencionado o inducido por contingencia de sede)
    target_proto = None
    proto_m = re.search(r"MC26-SEC-\d+", ql)
    if proto_m:
        target_proto = proto_m.group(0).upper()
    else:
        # Inducido por condición en sede (Puente de sede a protocolo)
        for cond, p in VENUE_OPS_PROTO:
            if cond in ql:
                target_proto = p
                break
                
    if target_proto and target_proto in PROTO_KB:
        info = PROTO_KB[target_proto]
        if any(w in ql for w in ["scope", "apply to", "cover", "encompass", "operations"]):
            return info["scope"]
        if any(w in ql for w in ["action", "step", "first", "initial", "opening"]):
            return info["action"]
        if any(w in ql for w in ["backup", "redundancy"]):
            return info["backup"]
        if any(w in ql for w in ["resumption", "resume", "restart", "sign off", "green light", "call", "restart"]):
            return info["resumption"]
            
    return None

# Evaluar en las 571 preguntas bridge de train
correct_bridge = 0
for b in [x for x in train_jsonl if x["type"] == "bridge"]:
    pred = solve_protocol_or_bridge(b["question"])
    gold = b["answer"].strip().lower()
    if pred == gold:
        correct_bridge += 1
    else:
        # print error
        pass

print(f"Acierto en puentes (bridge) de train: {correct_bridge}/571 ({correct_bridge/571*100:.2f}%)")
