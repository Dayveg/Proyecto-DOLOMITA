import json
import re
from pathlib import Path
from collections import Counter
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

retrieval_records = json.load(open("retrieval_index.json", encoding="utf-8"))
doc_chunk_map = {}
for r in retrieval_records:
    doc_chunk_map.setdefault(r["doc_id"], []).append(r["text"])

# Cargar index faiss
enc_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", local_files_only=True)
enc_model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", local_files_only=True).to("cuda")
enc_model.eval()

embeddings = np.array([r["vector"] for r in retrieval_records]).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def search_faiss(q, k=5):
    inputs = enc_tokenizer([q], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = enc_model(**inputs)
        vec = out[0][:, 0]
        vec = torch.nn.functional.normalize(vec, p=2, dim=1).cpu().numpy().astype("float32")
    scores, indices = index.search(vec, k)
    return [retrieval_records[i]["doc_id"] for i in indices[0]]

# Mapeos de dominio del problema DOLOMITA
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
    "milano": ["MC26-DOC-001", "MC26-DOC-002", "MC26-DOC-003", "MC26-DOC-004"],
    "cortina": ["MC26-DOC-005", "MC26-DOC-006", "MC26-DOC-007", "MC26-DOC-008"],
    "bormio": ["MC26-DOC-009", "MC26-DOC-010", "MC26-DOC-011", "MC26-DOC-012"],
    "livigno": ["MC26-DOC-013", "MC26-DOC-014", "MC26-DOC-015", "MC26-DOC-016"],
    "fiemme": ["MC26-DOC-017", "MC26-DOC-018", "MC26-DOC-019", "MC26-DOC-020"],
    "anterselva": ["MC26-DOC-021", "MC26-DOC-022", "MC26-DOC-023", "MC26-DOC-024"],
}

def retrieve_two_pass(query):
    # Detección de protocolos directos en la consulta
    found_protos = re.findall(r"MC26-SEC-\d+", query, flags=re.IGNORECASE)
    targeted_docs = set()
    for p in found_protos:
        p_up = p.upper()
        if p_up in PROTO_MAP:
            targeted_docs.add(PROTO_MAP[p_up])
            
    # Detección de comparación entre sedes
    ql = query.lower()
    venues_in_q = [v for v in VENUE_PREFIXES if v in ql]
    is_comparison = len(venues_in_q) >= 2 or "higher" in ql or "greater" in ql or "better" in ql
    
    # Pasada 1: FAISS
    top_docs = search_faiss(query, k=5)
    candidate_docs = set(top_docs[:3]) | targeted_docs
    
    # Pasada 2: Buscar puentes dentro del texto de los documentos candidatos (Hop 2)
    hop2_docs = set()
    for did in list(candidate_docs):
        doc_texts = " ".join(doc_chunk_map.get(did, []))
        
        # Buscar protocolos mencionados en doc1
        protos_in_doc = re.findall(r"MC26-SEC-\d+", doc_texts, flags=re.IGNORECASE)
        for p in protos_in_doc:
            p_up = p.upper()
            if p_up in PROTO_MAP:
                hop2_docs.add(PROTO_MAP[p_up])
                
        # Buscar roles clave mencionados en doc1
        doc_lower = doc_texts.lower()
        for role, charter_doc in ROLE_CHARTERS.items():
            if role in doc_lower and did != charter_doc:
                # Si la pregunta pregunta por reporte o escalamiento
                if any(w in ql for w in ["report", "boss", "escalat", "deadline", "answer to"]):
                    hop2_docs.add(charter_doc)
                    
    # Si es comparación entre dos sedes, incluir docs relevantes de ambas sedes
    if len(venues_in_q) == 2:
        for v in venues_in_q:
            # Los documentos 002, 006, 010, 014, 018, 022 son los de Operations and Thresholds donde están las tablas de comparación
            v_ops_doc = [d for d in VENUE_PREFIXES[v] if int(d.split("-")[-1]) % 4 == 2]
            for d in v_ops_doc:
                hop2_docs.add(d)

    final_docs = list(candidate_docs | hop2_docs)
    return final_docs

# Medir recall en train (support_doc_ids completamente contenidos en los docs recuperados)
hits = 0
single_hits = 0
bridge_hits = 0
comp_hits = 0

sample_train = train_jsonl[:500]
for x in sample_train:
    target_docs = set(x["support_doc_ids"])
    retrieved = set(retrieve_two_pass(x["question"]))
    ok = target_docs.issubset(retrieved)
    if ok:
        hits += 1
        if x["type"] == "single": single_hits += 1
        elif x["type"] == "bridge": bridge_hits += 1
        elif x["type"] == "comparison": comp_hits += 1

print(f"\nEvaluación en {len(sample_train)} preguntas:")
print(f"Recall global de soporte: {hits / len(sample_train) * 100:.2f}%")
print(f"Hits por tipo -> single: {single_hits}, bridge: {bridge_hits}, comparison: {comp_hits}")
