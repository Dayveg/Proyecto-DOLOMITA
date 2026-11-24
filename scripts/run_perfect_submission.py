import json
import re
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from scripts.normalizer import normalize_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo activo: {DEVICE}")

DATA_DIR = Path("data")
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

# Cargar FAISS
import faiss
import numpy as np

ENCODER_MODEL_ID = "BAAI/bge-base-en-v1.5"
enc_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID, local_files_only=True)
enc_model = AutoModel.from_pretrained(ENCODER_MODEL_ID, local_files_only=True).to(DEVICE)
enc_model.eval()

retrieval_records = json.load(open("retrieval_index.json", encoding="utf-8"))
embeddings = np.array([r["vector"] for r in retrieval_records]).astype("float32")
faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

PROTO_MAP = {f"MC26-SEC-{i:03d}": f"MC26-DOC-{24+i:03d}" for i in range(1, 21)}
ROLE_CHARTERS = {
    "chief of competition": "MC26-DOC-059",
    "venue manager": "MC26-DOC-060",
    "technical delegate": "MC26-DOC-061",
    "medical team lead": "MC26-DOC-062",
    "security lead": "MC26-DOC-063",
    "weather officer": "MC26-DOC-064",
    "logistics lead": "MC26-DOC-065",
}

VENUE_PREFIXES = {
    "milano": ("MC26-DOC-001", "MC26-DOC-002", "MC26-DOC-003", "MC26-DOC-004"),
    "cortina": ("MC26-DOC-005", "MC26-DOC-006", "MC26-DOC-007", "MC26-DOC-008"),
    "bormio": ("MC26-DOC-009", "MC26-DOC-010", "MC26-DOC-011", "MC26-DOC-012"),
    "livigno": ("MC26-DOC-013", "MC26-DOC-014", "MC26-DOC-015", "MC26-DOC-016"),
    "fiemme": ("MC26-DOC-017", "MC26-DOC-018", "MC26-DOC-019", "MC26-DOC-020"),
    "anterselva": ("MC26-DOC-021", "MC26-DOC-022", "MC26-DOC-023", "MC26-DOC-024"),
}

THRESHOLDS = {
    "snow": {"livigno": 50, "cortina": 45, "bormio": 40, "fiemme": 35, "anterselva": 30},
    "wind": {"bormio": 110, "cortina": 100, "fiemme": 90, "livigno": 80, "anterselva": 70},
    "visibility": {"fiemme": 300, "bormio": 250, "cortina": 200, "anterselva": 180, "livigno": 150},
    "temp": {"anterselva": -24, "bormio": -25, "livigno": -28, "cortina": -22, "fiemme": -20}
}

VENUES = ["milano", "cortina", "bormio", "livigno", "fiemme", "anterselva"]

def solve_comparison(q):
    ql = q.lower()
    found_venues = [v for v in VENUES if v in ql]
    if len(found_venues) != 2:
        return None
    v1, v2 = found_venues[0], found_venues[1]
    
    metric = None
    if any(w in ql for w in ["snow", "depth"]): metric = "snow"
    elif any(w in ql for w in ["wind", "speed"]): metric = "wind"
    elif any(w in ql for w in ["visib", "sight"]): metric = "visibility"
    elif any(w in ql for w in ["temp", "cold"]): metric = "temp"
    if not metric or metric not in THRESHOLDS: return None
    
    val1 = THRESHOLDS[metric].get(v1)
    val2 = THRESHOLDS[metric].get(v2)
    if val1 is None or val2 is None: return None
    
    is_lower = any(w in ql for w in ["lower", "less", "smaller", "below"])
    if is_lower:
        return v1 if val1 < val2 else v2
    else:
        return v1 if val1 > val2 else v2

def search_faiss(query: str, top_k: int = 3):
    with torch.no_grad():
        encoded = enc_tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        out = enc_model(**encoded)
        q_emb = out[0][:, 0]
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).cpu().numpy().astype("float32")
    scores, indices = faiss_index.search(q_emb, top_k)
    return [retrieval_records[idx]["doc_id"] for idx in indices[0]]

def get_targeted_context(query: str) -> str:
    ql = query.lower()
    ordered_docs = []
    
    # 1. Si menciona protocolo explícito
    for p, did in PROTO_MAP.items():
        if p.lower() in ql and did not in ordered_docs:
            ordered_docs.append(did)
            
    # 2. Si menciona sede
    venues_found = [v for v in VENUES if v in ql]
    if len(venues_found) == 1:
        v = venues_found[0]
        # Determinar si pregunta sobre umbrales/ops o staffing o systems
        is_ops = any(w in ql for w in ["wind", "snow", "temp", "visibility", "threshold", "minimum", "maximum", "cold", "speed", "depth"])
        is_staff = any(w in ql for w in ["who", "responsible", "lead", "boss", "report", "escalat", "deadline", "manage", "oversee", "access", "first aid", "timing calibration", "snow grooming", "safety"])
        is_sys = any(w in ql for w in ["backup", "power", "lighting", "generator", "ups", "reservoir", "system", "infrastructure"])
        
        # Poner en primer lugar el documento más exacto
        if is_ops:
            ordered_docs.extend([VENUE_PREFIXES[v][1], VENUE_PREFIXES[v][3], VENUE_PREFIXES[v][2]])
        elif is_staff:
            ordered_docs.extend([VENUE_PREFIXES[v][3], VENUE_PREFIXES[v][1], VENUE_PREFIXES[v][2]])
        elif is_sys:
            ordered_docs.extend([VENUE_PREFIXES[v][2], VENUE_PREFIXES[v][1], VENUE_PREFIXES[v][3]])
        else:
            ordered_docs.extend([VENUE_PREFIXES[v][1], VENUE_PREFIXES[v][3]])
            
    # 3. FAISS para complementar
    faiss_docs = search_faiss(query, top_k=3)
    for d in faiss_docs:
        if d not in ordered_docs:
            ordered_docs.append(d)
            
    # 4. Hop 2 (Bridges)
    hop2 = []
    for did in ordered_docs[:3]:
        txt = doc_text_map.get(did, "")
        for p, p_did in PROTO_MAP.items():
            if p in txt and p_did not in ordered_docs and p_did not in hop2:
                if any(w in ql for w in ["action", "step", "first", "restart", "resum", "protocol", "sign off", "green light", "hold", "who", "authority"]):
                    hop2.append(p_did)
        txt_l = txt.lower()
        if any(w in ql for w in ["report", "boss", "escalat", "deadline", "answer to"]):
            for role, charter_did in ROLE_CHARTERS.items():
                if role in txt_l and charter_did not in ordered_docs and charter_did not in hop2:
                    hop2.append(charter_did)
                    
    # Intercalar hop2 justo después de los primeros docs
    final_doc_list = []
    for d in (ordered_docs[:2] + hop2 + ordered_docs[2:]):
        if d not in final_doc_list:
            final_doc_list.append(d)
            
    # Tomar máximo 3 documentos completos sin cortar a la mitad
    top3_docs = final_doc_list[:3]
    return "\n\n".join([doc_text_map.get(d, "") for d in top3_docs])

PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query concisely with a short canonical phrase, role, value with unit, or protocol ID.
Query: {query}
Answer: """

def predict_answer(query: str) -> str:
    # A. Chequear si es comparación directa (100% exacto)
    comp_ans = solve_comparison(query)
    if comp_ans is not None:
        return comp_ans
        
    # B. Inferencia con LLM y contexto ordenado
    context = get_targeted_context(query)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    inputs = dec_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output_ids = dec_model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=dec_tokenizer.eos_token_id,
            eos_token_id=dec_tokenizer.eos_token_id
        )
        
    gen = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = dec_tokenizer.decode(gen, skip_special_tokens=True).strip().split("\n")[0].strip()
    return normalize_answer(raw)

print(f"Generando submission optimizado para {len(test_csv)} preguntas...")
predictions = []
for idx, row in test_csv.iterrows():
    pred = predict_answer(row["question"])
    predictions.append({"id": row["id"], "answer": pred})
    if (idx + 1) % 100 == 0 or (idx + 1) == len(test_csv):
        print(f"Procesadas {idx + 1}/{len(test_csv)} preguntas...")

sub_df = pd.DataFrame(predictions)
sub_path = Path("submissions/submission_v3_perfect_context.csv")
sub_df.to_csv(sub_path, index=False)
print(f"Guardado exitosamente en: {sub_path}")
