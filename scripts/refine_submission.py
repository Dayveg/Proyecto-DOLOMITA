import pandas as pd
import re

df_v3 = pd.read_csv("submissions/submission_v3_perfect_context.csv")
test_csv = pd.read_csv("data/test.csv")

# 1. Analizar T-00009, T-00026, etc. donde v3 puso erróneamente 'fire suppression system'
# En train vimos que cuando preguntan por backup de un protocolo específico:
PROTO_BACKUPS = {
    "MC26-SEC-001": "heated shelter",
    "MC26-SEC-002": "wind sensor array",
    "MC26-SEC-003": "fog lighting",
    "MC26-SEC-004": "snow additive",
    "MC26-SEC-005": "spare parts kit",
    "MC26-SEC-006": "diesel generator",
    "MC26-SEC-007": "ambulance",
    "MC26-SEC-008": "radio network",
    "MC26-SEC-009": "muster point",
    "MC26-SEC-010": "avalanche transceiver",
    "MC26-SEC-011": "ice resurfacer",
    "MC26-SEC-012": "radio network",
    "MC26-SEC-013": "barrier system",
    "MC26-SEC-014": "fire suppression system",
    "MC26-SEC-015": "shuttle fleet",
    "MC26-SEC-016": "chain-of-custody kit",
    "MC26-SEC-017": "barrier system",
    "MC26-SEC-018": "tracking beacon",
    "MC26-SEC-019": "counter-drone system",
    "MC26-SEC-020": "heated shelter"
}

# Corregir sobre el submission v3 (que ya tenía 0.88!)
new_answers = []
fixes = 0

for idx, r in test_csv.iterrows():
    qid = r["id"]
    q = r["question"]
    ql = q.lower()
    current_ans = df_v3.loc[df_v3["id"] == qid, "answer"].values[0]
    
    corrected = current_ans
    
    # Chequear si menciona un protocolo y pregunta por backup / equipo
    if any(w in ql for w in ["backup", "redundanc", "equipment"]):
        proto_m = re.search(r"MC26-SEC-\d+", ql)
        if proto_m:
            proto = proto_m.group(0).upper()
            if proto in PROTO_BACKUPS and proto != "MC26-SEC-014":
                expected_bk = PROTO_BACKUPS[proto]
                if current_ans != expected_bk:
                    corrected = expected_bk
                    fixes += 1
                    
    # Chequear si pregunta por la primera acción / step 1 de un protocolo
    PROTO_ACTIONS = {
        "MC26-SEC-001": "suspend outdoor activity",
        "MC26-SEC-002": "hold the competition",
        "MC26-SEC-003": "delay the start",
        "MC26-SEC-004": "activate snowmaking",
        "MC26-SEC-005": "switch to backup system",
        "MC26-SEC-006": "start the diesel generator",
        "MC26-SEC-007": "dispatch the medical team",
        "MC26-SEC-008": "lock down the venue",
        "MC26-SEC-009": "muster point",
        "MC26-SEC-010": "close the slope",
        "MC26-SEC-011": "clear the ice surface",
        "MC26-SEC-012": "switch to the radio network",
        "MC26-SEC-013": "open additional exits",
        "MC26-SEC-014": "activate the fire suppression system",
        "MC26-SEC-015": "reroute the shuttle fleet",
        "MC26-SEC-016": "escort the athlete to the control station",
        "MC26-SEC-017": "raise the security level",
        "MC26-SEC-018": "alert the venue manager",
        "MC26-SEC-019": "activate the counter-drone system",
        "MC26-SEC-020": "move personnel to shelter"
    }
    if any(w in ql for w in ["action", "step", "first", "initial", "opening"]):
        proto_m = re.search(r"MC26-SEC-\d+", ql)
        if proto_m:
            proto = proto_m.group(0).upper()
            if proto in PROTO_ACTIONS:
                expected_act = PROTO_ACTIONS[proto]
                if current_ans != expected_act:
                    corrected = expected_act
                    fixes += 1

    # Chequear si pregunta por el scope de un protocolo
    PROTO_SCOPES = {
        "MC26-SEC-001": "venue only", "MC26-SEC-002": "games wide", "MC26-SEC-003": "venue only", "MC26-SEC-004": "games wide",
        "MC26-SEC-005": "venue only", "MC26-SEC-006": "games wide", "MC26-SEC-007": "venue only", "MC26-SEC-008": "games wide",
        "MC26-SEC-009": "venue only", "MC26-SEC-010": "games wide", "MC26-SEC-011": "venue only", "MC26-SEC-012": "games wide",
        "MC26-SEC-013": "venue only", "MC26-SEC-014": "games wide", "MC26-SEC-015": "venue only", "MC26-SEC-016": "games wide",
        "MC26-SEC-017": "venue only", "MC26-SEC-018": "games wide", "MC26-SEC-019": "venue only", "MC26-SEC-020": "games wide",
    }
    if any(w in ql for w in ["scope", "apply", "cover", "encompass", "operations", "functions"]):
        proto_m = re.search(r"MC26-SEC-\d+", ql)
        if proto_m:
            proto = proto_m.group(0).upper()
            if proto in PROTO_SCOPES:
                expected_sc = PROTO_SCOPES[proto]
                if current_ans != expected_sc:
                    corrected = expected_sc
                    fixes += 1

    # Chequear si pregunta por resumption de un protocolo
    PROTO_RESUMPTION = {
        "MC26-SEC-001": "chief of competition", "MC26-SEC-002": "venue manager", "MC26-SEC-003": "technical delegate",
        "MC26-SEC-004": "medical team lead", "MC26-SEC-005": "security lead", "MC26-SEC-006": "weather officer",
        "MC26-SEC-007": "logistics lead", "MC26-SEC-008": "chief of competition", "MC26-SEC-009": "venue manager",
        "MC26-SEC-010": "technical delegate", "MC26-SEC-011": "medical team lead", "MC26-SEC-012": "security lead",
        "MC26-SEC-013": "weather officer", "MC26-SEC-014": "logistics lead", "MC26-SEC-015": "chief of competition",
        "MC26-SEC-016": "venue manager", "MC26-SEC-017": "technical delegate", "MC26-SEC-018": "medical team lead",
        "MC26-SEC-019": "security lead", "MC26-SEC-020": "weather officer"
    }
    if any(w in ql for w in ["resumption", "resume", "restart", "sign off", "green light", "call", "authoriz"]):
        proto_m = re.search(r"MC26-SEC-\d+", ql)
        if proto_m:
            proto = proto_m.group(0).upper()
            if proto in PROTO_RESUMPTION:
                expected_res = PROTO_RESUMPTION[proto]
                if current_ans != expected_res:
                    corrected = expected_res
                    fixes += 1

    new_answers.append(corrected)

print(f"Total correcciones de precisión sobre submission v3: {fixes}")

df_fixed = df_v3.copy()
df_fixed["answer"] = new_answers
out_p = "submissions/submission_v5_elite.csv"
df_fixed.to_csv(out_p, index=False)
print(f"Guardado exitosamente en: {out_p}")
