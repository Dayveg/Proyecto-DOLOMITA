import json
from pathlib import Path
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
from scripts.chunker import get_all_chunks

def build_and_save_index():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Indexador usando dispositivo: {device}")
    
    kb_dir = Path("data/base_conocimiento")
    chunks = get_all_chunks(kb_dir)
    print(f"Indexando {len(chunks)} chunks...")
    
    encoder_name = "BAAI/bge-base-en-v1.5"
    print(f"Cargando encoder {encoder_name}...")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = AutoModel.from_pretrained(encoder_name).to(device)
    model.eval()
    
    texts = [c["text"] for c in chunks]
    batch_size = 64
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = model(**encoded)
            # CLS pooling y normalización L2
            cls_embeddings = outputs[0][:, 0]
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
            all_embeddings.append(cls_embeddings.cpu().numpy())
            
    embeddings = np.vstack(all_embeddings).astype("float32")
    dim = embeddings.shape[1]
    print(f"Embeddings generados: {embeddings.shape}")
    
    # Índice FAISS inner product (equivalente a cosine similarity por estar normalizados)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Guardar índice FAISS en disco
    faiss_path = Path("models/faiss_index.bin")
    faiss.write_index(index, str(faiss_path))
    print(f"Índice FAISS guardado en {faiss_path}")
    
    # Guardar retrieval_index.json con la estructura oficial requerida
    # [ {"doc_id": "...", "chunk_id": 0, "text": "...", "vector": [...]}, ... ]
    retrieval_records = []
    for c, vec in zip(chunks, embeddings):
        retrieval_records.append({
            "doc_id": c["doc_id"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "vector": [round(float(v), 5) for v in vec]
        })
        
    index_json_path = Path("retrieval_index.json")
    with open(index_json_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_records, f)
    print(f"retrieval_index.json guardado en {index_json_path} con {len(retrieval_records)} registros.")

if __name__ == "__main__":
    build_and_save_index()
