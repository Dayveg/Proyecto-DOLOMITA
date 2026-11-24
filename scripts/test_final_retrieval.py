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

KB_DIR = DATA_DIR / "base_conocimiento"
doc_texts = {p.parent.name: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*/doc.md"))}

retrieval_records = json.load(open("retrieval_index.json", encoding="utf-8"))

enc_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", local_files_only=True)
enc_model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", local_files_only=True).to("cuda")
enc_model.eval()

embeddings = np.array([r["vector"] for r in retrieval_records]).astype("float32")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

def search_faiss(q, k=4):
    inputs = enc_tokenizer([q], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = enc_model(**inputs)
        vec = out[0][:, 0]
        vec = torch.nn.functional.normalize(vec, p=2, dim=1).cpu().numpy().astype("float32")
    scores, indices = index.search(vec, k)
    return [retrieval_records[i]["doc_id"] for i in indices[0]]

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

def retrieve_multi_hop(query):
    ql = query.lower()
    selected_docs = set()
    
    # 1. Chequeo de protocolo explícito
    for p, did in PROTO_MAP.items():
        if p.lower() in ql:
            selected_docs.add(did)
            
    # 2. Chequeo de sedes
    venues_found = [v for v in VENUE_PREFIXES if v in ql]
    
    if len(venues_found) >= 2:
        # Comparison: traer Ops & Thresholds de ambas
        for v in venues_found:
            selected_docs.add(VENUE_PREFIXES[v][1])
            selected_docs.add(VENUE_PREFIXES[v][0])
            
    elif len(venues_found) == 1:
        v = venues_found[0]
        # Agregar los docs relevantes de la sede
        selected_docs.add(VENUE_PREFIXES[v][1]) # ops
        selected_docs.add(VENUE_PREFIXES[v][3]) # staffing
        selected_docs.add(VENUE_PREFIXES[v][2]) # systems
        
    # 3. FAISS Retrieval para complementar
    faiss_top = search_faiss(query, k=3)
    selected_docs.update(faiss_top)
    
    # 4. SALTO 2 (HOP 2): Explorar puentes en el texto de los documentos recuperados
    hop2_docs = set()
    for did in list(selected_docs):
        txt = doc_texts.get(did, "")
        
        # Puente 1: Menciona protocolos MC26-SEC-XXX
        for p, p_did in PROTO_MAP.items():
            if p in txt and p_did != did:
                # Si la pregunta sugiere acción, contingencia o protocolo
                if any(w in ql for w in ["action", "step", "first", "restart", "resum", "protocol", "sign off", "green light", "hold", "who", "authority"]):
                    hop2_docs.add(p_did)
                    
        # Puente 2: Menciona roles y la pregunta busca reporte o escalamiento
        txt_lower = txt.lower()
        if any(w in ql for w in ["report", "boss", "escalat", "deadline", "answer to"]):
            for role, charter_did in ROLE_CHARTERS.items():
                if role in txt_lower and charter_did != did:
                    hop2_docs.add(charter_did)
                    
    selected_docs.update(hop2_docs)
    return selected_docs

# Medir cobertura sobre TODO train
hits = 0
by_type_hits = Counter()

for x in train_jsonl:
    targets = set(x["support_doc_ids"])
    retrieved = retrieve_multi_hop(x["question"])
    if targets.issubset(retrieved):
        hits += 1
        by_type_hits[x["type"]] += 1

print(f"Total evaluados: {len(train_jsonl)}")
print(f"Recall Global Multi-Hop: {hits}/{len(train_jsonl)} ({hits/len(train_jsonl)*100:.2f}%)")
print("Hits por tipo:", by_type_hits)
print(f"Total fallos: {len(train_jsonl) - hits}")
