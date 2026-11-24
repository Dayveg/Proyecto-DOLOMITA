import re
import json
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel

# Importar funciones de scripts
from scripts.normalizer import normalize_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo activo: {DEVICE}")

DATA_DIR = Path("data")
test_csv = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)

KB_DIR = DATA_DIR / "base_conocimiento"
doc_text_map = {p.parent.name: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*/doc.md"))}

# Cargar encoder
ENCODER_MODEL_ID = "BAAI/bge-base-en-v1.5"
enc_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID, local_files_only=True)
enc_model = AutoModel.from_pretrained(ENCODER_MODEL_ID, local_files_only=True).to(DEVICE)
enc_model.eval()

# Cargar faiss index
import faiss
import numpy as np

retrieval_records = json.load(open("retrieval_index.json", encoding="utf-8"))
embeddings = np.array([r["vector"] for r in retrieval_records]).astype("float32")
faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
faiss_index.add(embeddings)

# Cargar decoder fine-tuneado
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

PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query concisely with a short canonical phrase, role, value with unit, or protocol ID.
Query: {query}
Answer: """

def search_faiss(query: str, top_k: int = 3):
    with torch.no_grad():
        encoded = enc_tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        out = enc_model(**encoded)
        q_emb = out[0][:, 0]
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).cpu().numpy().astype("float32")
    scores, indices = faiss_index.search(q_emb, top_k)
    return [retrieval_records[idx]["doc_id"] for idx in indices[0]]

def retrieve_multi_hop_docs(query: str):
    ql = query.lower()
    selected_docs = set()
    
    # 1. Chequeo de protocolo explícito
    for p, did in PROTO_MAP.items():
        if p.lower() in ql:
            selected_docs.add(did)
            
    # 2. Chequeo de sedes
    venues_found = [v for v in VENUE_PREFIXES if v in ql]
    if len(venues_found) >= 2:
        for v in venues_found:
            selected_docs.add(VENUE_PREFIXES[v][1])
    elif len(venues_found) == 1:
        v = venues_found[0]
        selected_docs.add(VENUE_PREFIXES[v][1])
        selected_docs.add(VENUE_PREFIXES[v][3])
        selected_docs.add(VENUE_PREFIXES[v][2])
        
    # 3. FAISS Retrieval densa
    faiss_docs = search_faiss(query, top_k=3)
    selected_docs.update(faiss_docs)
    
    # 4. SALTO 2 (HOP 2): Búsqueda de puentes (Bridge)
    hop2_docs = set()
    for did in list(selected_docs):
        txt = doc_text_map.get(did, "")
        for p, p_did in PROTO_MAP.items():
            if p in txt and p_did != did:
                if any(w in ql for w in ["action", "step", "first", "restart", "resum", "protocol", "sign off", "green light", "hold", "who", "authority"]):
                    hop2_docs.add(p_did)
        txt_lower = txt.lower()
        if any(w in ql for w in ["report", "boss", "escalat", "deadline", "answer to"]):
            for role, charter_did in ROLE_CHARTERS.items():
                if role in txt_lower and charter_did != did:
                    hop2_docs.add(charter_did)
                    
    selected_docs.update(hop2_docs)
    return list(selected_docs)

def predict_answer(query: str) -> str:
    doc_ids = retrieve_multi_hop_docs(query)
    context_parts = [doc_text_map.get(did, "") for did in doc_ids]
    context = "\n\n".join(context_parts)[:3500]
    
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
        
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_ans = dec_tokenizer.decode(generated, skip_special_tokens=True).strip()
    raw_ans = raw_ans.split("\n")[0].strip()
    return normalize_answer(raw_ans)

print(f"Generando submission con Retrieval Multi-Hop para {len(test_csv)} preguntas...")
predictions = []
for idx, row in test_csv.iterrows():
    pred = predict_answer(row["question"])
    predictions.append({"id": row["id"], "answer": pred})
    if (idx + 1) % 100 == 0 or (idx + 1) == len(test_csv):
        print(f"Procesadas {idx + 1}/{len(test_csv)} preguntas...")

sub_df = pd.DataFrame(predictions)
sub_path = Path("submissions/submission_v2_multihop.csv")
sub_df.to_csv(sub_path, index=False)
print(f"Guardado en {sub_path} exitosamente.")
