import json
import re
from pathlib import Path
from collections import Counter
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import scripts.test_routing as tr

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

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

# Combinación híbrida: Routing de Dominio (para multi-hop, comparación y protocolos) + Top-K FAISS
hits = 0
by_type_hits = Counter()

for x in train_jsonl:
    targets = set(x["support_doc_ids"])
    # 1. Routing
    routed = tr.perfect_routing(x["question"])
    # 2. FAISS
    faiss_docs = search_faiss(x["question"], k=3)
    
    combined = set(routed) | set(faiss_docs)
    if targets.issubset(combined):
        hits += 1
        by_type_hits[x["type"]] += 1

print(f"Total evaluados: {len(train_jsonl)}")
print(f"Cobertura Híbrida (Routing + FAISS): {hits}/{len(train_jsonl)} ({hits/len(train_jsonl)*100:.2f}%)")
print("Hits por tipo:", by_type_hits)
print(f"Total fallos: {len(train_jsonl) - hits}")
