import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from scripts.normalizer import normalize_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo activo: {DEVICE}")

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

test_csv = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)

KB_DIR = DATA_DIR / "base_conocimiento"
doc_text_map = {p.parent.name: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*/doc.md"))}

# Cargar LLM
DECODER_MODEL_ID = "Qwen/Qwen2.5-1.5B"
dec_tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL_ID, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    DECODER_MODEL_ID,
    local_files_only=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
dec_model = PeftModel.from_pretrained(base_model, "models/decoder")
dec_model.eval()

# 1. Comparaciones (100% exacto en train)
from scripts.test_comparison_solver import solve_comparison

# 2. Solucionador de puentes y protocolos determinista
from scripts.test_bridge_solver import solve_protocol_or_bridge

# 3. Router de contexto enfocado para el LLM en preguntas complejas/single
VENUE_PREFIXES = {
    "milano": ("MC26-DOC-001", "MC26-DOC-002", "MC26-DOC-003", "MC26-DOC-004"),
    "cortina": ("MC26-DOC-005", "MC26-DOC-006", "MC26-DOC-007", "MC26-DOC-008"),
    "bormio": ("MC26-DOC-009", "MC26-DOC-010", "MC26-DOC-011", "MC26-DOC-012"),
    "livigno": ("MC26-DOC-013", "MC26-DOC-014", "MC26-DOC-015", "MC26-DOC-016"),
    "fiemme": ("MC26-DOC-017", "MC26-DOC-018", "MC26-DOC-019", "MC26-DOC-020"),
    "anterselva": ("MC26-DOC-021", "MC26-DOC-022", "MC26-DOC-023", "MC26-DOC-024"),
}
PROTO_MAP = {f"MC26-SEC-{i:03d}": f"MC26-DOC-{24+i:03d}" for i in range(1, 21)}

def get_clean_doc(query: str):
    ql = query.lower()
    for p, did in PROTO_MAP.items():
        if p.lower() in ql:
            return doc_text_map.get(did, "")
    for v, docs in VENUE_PREFIXES.items():
        if v in ql:
            if any(w in ql for w in ["wind", "snow", "temp", "visibility", "cutoff", "threshold", "minimum", "maximum"]):
                return doc_text_map.get(docs[1], "")
            if any(w in ql for w in ["who", "responsible", "lead", "boss", "manage", "oversee", "first aid", "access", "timing"]):
                return doc_text_map.get(docs[3], "")
            if any(w in ql for w in ["backup", "power", "lighting", "ups", "generator", "reservoir"]):
                return doc_text_map.get(docs[2], "")
            return doc_text_map.get(docs[1], "") + "\n\n" + doc_text_map.get(docs[3], "")
    return ""

PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query concisely with a short canonical phrase, role, value with unit, or protocol ID.
Query: {query}
Answer: """

def predict_master(q: str) -> str:
    # A. Comparación directa (100% de acierto)
    ans = solve_comparison(q)
    if ans:
        return normalize_answer(ans)
        
    # B. Puente / Protocolo determinista
    ans = solve_protocol_or_bridge(q)
    if ans:
        return normalize_answer(ans)
        
    # C. LLM con el documento exacto de la sede o protocolo
    ctx = get_clean_doc(q)
    prompt = PROMPT_TEMPLATE.format(context=ctx[:2500], query=q)
    inputs = dec_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = dec_model.generate(**inputs, max_new_tokens=25, do_sample=False, pad_token_id=dec_tokenizer.eos_token_id, eos_token_id=dec_tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = dec_tokenizer.decode(gen, skip_special_tokens=True).strip().split("\n")[0].strip()
    return normalize_answer(raw)

# Evaluar en 300 preguntas de train
sample = train_jsonl[:300]
hits = 0
for x in sample:
    p = predict_master(x["question"])
    g = normalize_answer(x["answer"])
    if p == g:
        hits += 1

print(f"\nExact Match en 300 preguntas de train: {hits}/300 ({hits/300*100:.2f}%)")

# Generar submission final
print(f"\nGenerando submission de alta precisión para {len(test_csv)} preguntas...")
preds = []
for idx, r in test_csv.iterrows():
    p = predict_master(r["question"])
    preds.append({"id": r["id"], "answer": p})

sub_df = pd.DataFrame(preds)
out_path = Path("submissions/submission_v4_master.csv")
sub_df.to_csv(out_path, index=False)
print(f"Archivo guardado en {out_path} exitosamente.")
