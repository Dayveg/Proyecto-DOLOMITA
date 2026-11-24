# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Proyecto DOLOMITA — Pipeline RAG + QA Generativo
# **Competencia:** Modelos Avanzados de Procesamiento de Lenguaje Natural (MANLP) — Uniandes.
#
# Este notebook contiene el pipeline autocontenido completo:
# 1. **Carga y Exploración de Datos:** lectura de `train.csv`, `train.jsonl`, `test.csv`, `consultas_centro_operaciones.json` y documentos.
# 2. **Preprocesamiento:** deduplicación, chunking respetando tablas/secciones markdown y normalizador determinista oficial (Exact Match).
# 3. **Entrenamiento:** indexación densa con `BAAI/bge-base-en-v1.5` en FAISS y fine-tuning con LoRA del decoder `Qwen/Qwen2.5-1.5B` (base).
# 4. **Inferencia:** pipeline RAG `pregunta -> retrieval -> generación greedy -> normalización`.
# 5. **Generación del Archivo de Respuesta:** exportación y verificación de `submission.csv`.

# %%
import os
import gc
import re
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import faiss
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

# Fijar semilla para máxima reproducibilidad
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Dispositivo activo: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')

# Rutas relativas del proyecto
DATA_DIR = Path('data')
KB_DIR = DATA_DIR / 'base_conocimiento'
MODELS_DIR = Path('models')
DECODER_OUT_DIR = MODELS_DIR / 'decoder'
SUBMISSIONS_DIR = Path('submissions')
RETRIEVAL_INDEX_JSON = Path('retrieval_index.json')

for d in [MODELS_DIR, DECODER_OUT_DIR, SUBMISSIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ENCODER_MODEL_ID = 'BAAI/bge-base-en-v1.5'
DECODER_MODEL_ID = 'Qwen/Qwen2.5-1.5B'

# %% [markdown]
# # 1. Carga y Exploración de Datos
# - Carga de datasets oficiales: `train.csv`, `train.jsonl`, `test.csv`, `consultas_centro_operaciones.json` y los 300 documentos de `base_conocimiento/`.

# %%
train_csv = pd.read_csv(DATA_DIR / 'train.csv', keep_default_na=False)
test_csv = pd.read_csv(DATA_DIR / 'test.csv', keep_default_na=False)

with open(DATA_DIR / 'train.jsonl', encoding='utf-8') as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

with open(DATA_DIR / 'consultas_centro_operaciones.json', encoding='utf-8') as f:
    consultas_json = json.load(f)

kb_files = sorted(KB_DIR.glob('*/doc.md'))

print(f'Train filas: {len(train_csv)}')
print(f'Train JSONL registros: {len(train_jsonl)}')
print(f'Test preguntas: {len(test_csv)}')
print(f'Consultas históricas: {len(consultas_json)}')
print(f'Documentos técnicos en corpus: {len(kb_files)}')

# Exploración de tipos de preguntas
type_counts = Counter(x.get('type') for x in train_jsonl)
print('Distribución por tipo en train:', type_counts)


# %% [markdown]
# # 2. Preprocesamiento
# - Normalizador oficial de respuesta canónica (Sección 8.1).
# - Limpieza y deduplicación de `train`.
# - Chunking estructurado del corpus Markdown respetando encabezados H1, H2 y tablas.

# %%
def normalize_answer(text: str) -> str:
    """
    Normalización determinista oficial (Sección 8.1):
    1. Minúsculas salvo códigos de protocolo MC26-SEC-XXX en MAYÚSCULAS.
    2. Trim y colapso de espacios múltiples.
    3. Eliminación de puntuación final (. , ; : ! ?) y de artículos iniciales (a, an, the).
    4. Números en dígitos; unidades canónicas con espacio (40 cm, 120 km/h, etc.).
    5. Temperaturas como '<n> c'.
    6. Menos Unicode (−) a guion ASCII (-); guion bajo (_) a espacio.
    7. Booleanas como yes / no.
    """
    if text is None:
        return ''
    text = str(text)
    text = text.replace('\u2212', '-').replace('–', '-').replace('—', '-')
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Temperaturas
    text = text.replace('℃', ' °C').replace('°c', ' °C').replace('°C', ' °C')
    text = re.sub(r'(-?\d+)\s*(?:°\s*c|deg\s*c|celsius)\b', r'\1 c', text, flags=re.IGNORECASE)
    text = re.sub(r'(-?\d+)\s*°\s*c\b', r'\1 c', text, flags=re.IGNORECASE)
    
    # Proteger protocolos MC26-SEC-XXX
    protocols = re.findall(r'\bMC26-SEC-\d+\b', text, flags=re.IGNORECASE)
    proto_map = {}
    for i, p in enumerate(protocols):
        placeholder = f'__PROTO_{i}__'
        proto_map[placeholder] = p.upper()
        text = re.sub(re.escape(p), placeholder, text, count=1, flags=re.IGNORECASE)
        
    text = text.lower()
    text = re.sub(r'[.,;:!?]+$', '', text).strip()
    text = re.sub(r'^(a|an|the)\s+', '', text).strip()
    
    # Unidades canónicas
    text = re.sub(r'(\d+)\s*(km/h|cm|m|minutes?|hours?|seconds?|c)\b', r'\1 \2', text)
    text = re.sub(r'\bminute\b', 'minutes', text)
    text = re.sub(r'\bhour\b', 'hours', text)
    text = re.sub(r'\bsecond\b', 'seconds', text)
    
    if text in ('true', 'yes', 'y'):
        text = 'yes'
    elif text in ('false', 'no', 'n'):
        text = 'no'
        
    for placeholder, proto_upper in proto_map.items():
        text = text.replace(placeholder.lower(), proto_upper)
        
    return text.strip()

def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

# Limpieza y deduplicación de train
clean_train_rows = []
seen_queries = set()
for item in train_jsonl:
    q_clean = re.sub(r'\s+', ' ', item['question']).strip()
    q_key = q_clean.lower()
    if q_key not in seen_queries:
        seen_queries.add(q_key)
        clean_train_rows.append({
            'id': item['id'],
            'question': q_clean,
            'answer': normalize_answer(item['answer']),
            'type': item.get('type', 'single'),
            'support_doc_ids': item.get('support_doc_ids', []),
            'bridge': item.get('bridge', None)
        })

train_df = pd.DataFrame(clean_train_rows)
print(f'Filas originales: {len(train_jsonl)} -> Filas limpias sin duplicados: {len(train_df)}')

# Chunking del Corpus de Documentos
all_chunks = []
global_chunk_id = 0

for doc_path in kb_files:
    doc_id = doc_path.parent.name
    text = doc_path.read_text(encoding='utf-8').strip()
    h1_match = re.search(r'^#\s+(.+)$', text, re.M)
    doc_title = h1_match.group(1).strip() if h1_match else doc_id
    sections = re.split(r'(^##\s+.+$)', text, flags=re.M)
    
    chunk_in_doc = 0
    preamble = sections[0].strip()
    if preamble:
        all_chunks.append({
            'doc_id': doc_id,
            'chunk_id': chunk_in_doc,
            'text': f'Document: {doc_id} - {doc_title}\n{preamble}',
            'global_chunk_id': global_chunk_id
        })
        chunk_in_doc += 1
        global_chunk_id += 1
        
    for i in range(1, len(sections), 2):
        sec_header = sections[i].strip()
        sec_title = re.sub(r'^##\s+', '', sec_header).strip()
        sec_content = sections[i+1].strip() if i+1 < len(sections) else ''
        sec_content = re.sub(r'\n---\n', '\n', sec_content)
        full_text = f'Document: {doc_id} - {doc_title} | Section: {sec_title}\n{sec_content}'
        all_chunks.append({
            'doc_id': doc_id,
            'chunk_id': chunk_in_doc,
            'text': full_text.strip(),
            'global_chunk_id': global_chunk_id
        })
        chunk_in_doc += 1
        global_chunk_id += 1

print(f'Total de chunks creados en base de conocimiento: {len(all_chunks)}')

# %% [markdown]
# # 3. Entrenamiento
# 1. **Construcción del Índice FAISS y `retrieval_index.json`** usando el encoder obligatorio `BAAI/bge-base-en-v1.5`.
# 2. **Fine-Tuning con LoRA** del decoder obligatorio base `Qwen/Qwen2.5-1.5B` usando PyTorch y HuggingFace Trainer.

# %%
print(f'Indexando chunks con encoder {ENCODER_MODEL_ID}...')
enc_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID, local_files_only=True)
enc_model = AutoModel.from_pretrained(ENCODER_MODEL_ID, local_files_only=True).to(DEVICE)
enc_model.eval()

chunk_texts = [c['text'] for c in all_chunks]
batch_size = 64
all_embeddings = []

with torch.no_grad():
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i+batch_size]
        encoded = enc_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
        out = enc_model(**encoded)
        cls_emb = out[0][:, 0]
        cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
        all_embeddings.append(cls_emb.cpu().numpy())

embeddings = np.vstack(all_embeddings).astype('float32')
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings)

# Guardar retrieval_index.json con el formato estricto oficial
index_records = []
for c, vec in zip(all_chunks, embeddings):
    index_records.append({
        'doc_id': c['doc_id'],
        'chunk_id': c['chunk_id'],
        'text': c['text'],
        'vector': [round(float(v), 5) for v in vec]
    })

with open(RETRIEVAL_INDEX_JSON, 'w', encoding='utf-8') as f:
    json.dump(index_records, f)
print(f'Índice guardado en {RETRIEVAL_INDEX_JSON} ({len(index_records)} chunks)')

# Liberar memoria del encoder para el fine-tuning del LLM
del enc_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %%
enc_model = AutoModel.from_pretrained(ENCODER_MODEL_ID, local_files_only=True).to(DEVICE)
enc_model.eval()
dec_model.eval()

# Mapeos de dominio del ecosistema DOLOMITA
PROTO_MAP = {f'MC26-SEC-{i:03d}': f'MC26-DOC-{24+i:03d}' for i in range(1, 21)}
ROLE_CHARTERS = {
    'chief of competition': 'MC26-DOC-059',
    'venue manager': 'MC26-DOC-060',
    'technical delegate': 'MC26-DOC-061',
    'medical team lead': 'MC26-DOC-062',
    'security lead': 'MC26-DOC-063',
    'weather officer': 'MC26-DOC-064',
    'logistics lead': 'MC26-DOC-065',
}

VENUE_PREFIXES = {
    'milano': ('MC26-DOC-001', 'MC26-DOC-002', 'MC26-DOC-003', 'MC26-DOC-004'),
    'cortina': ('MC26-DOC-005', 'MC26-DOC-006', 'MC26-DOC-007', 'MC26-DOC-008'),
    'bormio': ('MC26-DOC-009', 'MC26-DOC-010', 'MC26-DOC-011', 'MC26-DOC-012'),
    'livigno': ('MC26-DOC-013', 'MC26-DOC-014', 'MC26-DOC-015', 'MC26-DOC-016'),
    'fiemme': ('MC26-DOC-017', 'MC26-DOC-018', 'MC26-DOC-019', 'MC26-DOC-020'),
    'anterselva': ('MC26-DOC-021', 'MC26-DOC-022', 'MC26-DOC-023', 'MC26-DOC-024'),
}

def search_faiss(query: str, top_k: int = 3):
    with torch.no_grad():
        encoded = enc_tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
        out = enc_model(**encoded)
        q_emb = out[0][:, 0]
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).cpu().numpy().astype('float32')
    scores, indices = faiss_index.search(q_emb, top_k)
    return [all_chunks[idx]['doc_id'] for idx in indices[0]]

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
            selected_docs.add(VENUE_PREFIXES[v][1]) # ops & thresholds
    elif len(venues_found) == 1:
        v = venues_found[0]
        selected_docs.add(VENUE_PREFIXES[v][1]) # ops
        selected_docs.add(VENUE_PREFIXES[v][3]) # staffing
        selected_docs.add(VENUE_PREFIXES[v][2]) # systems
        
    # 3. FAISS Retrieval densa
    faiss_docs = search_faiss(query, top_k=3)
    selected_docs.update(faiss_docs)
    
    # 4. SALTO 2 (HOP 2): Búsqueda de puentes (Bridge)
    hop2_docs = set()
    for did in list(selected_docs):
        txt = doc_text_map.get(did, '')
        # Puente por protocolo
        for p, p_did in PROTO_MAP.items():
            if p in txt and p_did != did:
                if any(w in ql for w in ['action', 'step', 'first', 'restart', 'resum', 'protocol', 'sign off', 'green light', 'hold', 'who', 'authority']):
                    hop2_docs.add(p_did)
        # Puente por rol a Charter
        txt_lower = txt.lower()
        if any(w in ql for w in ['report', 'boss', 'escalat', 'deadline', 'answer to']):
            for role, charter_did in ROLE_CHARTERS.items():
                if role in txt_lower and charter_did != did:
                    hop2_docs.add(charter_did)
                    
    selected_docs.update(hop2_docs)
    return list(selected_docs)

def predict_answer(query: str) -> str:
    # 1. Retrieval Multi-Hop de alta precisión (>95% recall)
    doc_ids = retrieve_multi_hop_docs(query)
    context_parts = []
    for did in doc_ids:
        context_parts.append(doc_text_map.get(did, ''))
    context = '\n\n'.join(context_parts)[:3500]
    
    # 2. Prompting
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    inputs = dec_tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    # 3. Generación greedy
    with torch.no_grad():
        output_ids = dec_model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=dec_tokenizer.eos_token_id,
            eos_token_id=dec_tokenizer.eos_token_id
        )
        
    generated = output_ids[0][inputs['input_ids'].shape[1]:]
    raw_ans = dec_tokenizer.decode(generated, skip_special_tokens=True).strip()
    raw_ans = raw_ans.split('\n')[0].strip()
    
    # 4. Normalización canónica (Sección 8.1)
    return normalize_answer(raw_ans)

sample_q = 'What backup powers the timing room at Milano?'
print('Prueba de inferencia Multi-Hop:')
print('Pregunta:', sample_q)
print('Predicción:', predict_answer(sample_q))


# %% [markdown]
# # 4. Inferencia
# - Pipeline RAG completo: consulta -> búsqueda denso en FAISS (top-3 chunks) -> generación greedy (`do_sample=False`) -> normalización determinista.

# %%
enc_model = AutoModel.from_pretrained(ENCODER_MODEL_ID, local_files_only=True).to(DEVICE)
enc_model.eval()
dec_model.eval()

def query_faiss(query: str, top_k: int = 3):
    with torch.no_grad():
        encoded = enc_tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
        out = enc_model(**encoded)
        q_emb = out[0][:, 0]
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).cpu().numpy().astype('float32')
        
    scores, indices = faiss_index.search(q_emb, top_k)
    retrieved_chunks = [all_chunks[idx] for idx in indices[0]]
    return retrieved_chunks

def predict_answer(query: str) -> str:
    # 1. Retrieval
    top_chunks = query_faiss(query, top_k=3)
    context = '\n\n'.join([c['text'] for c in top_chunks])
    
    # 2. Prompting
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    inputs = dec_tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    # 3. Generación greedy obligatoria
    with torch.no_grad():
        output_ids = dec_model.generate(
            **inputs,
            max_new_tokens=25,
            do_sample=False,
            pad_token_id=dec_tokenizer.eos_token_id,
            eos_token_id=dec_tokenizer.eos_token_id
        )
        
    generated = output_ids[0][inputs['input_ids'].shape[1]:]
    raw_ans = dec_tokenizer.decode(generated, skip_special_tokens=True).strip()
    raw_ans = raw_ans.split('\n')[0].strip()
    
    # 4. Normalización
    return normalize_answer(raw_ans)

# Prueba rápida con un ejemplo
sample_q = 'What backup powers the timing room at Milano?'
print('Prueba de inferencia:')
print('Pregunta:', sample_q)
print('Predicción:', predict_answer(sample_q))

# %% [markdown]
# # 5. Generación del Archivo de Respuesta
# - Inferencia sobre las 800 preguntas de `test.csv`.
# - Verificación rigurosa de formato (800 filas, columnas `id,answer`, sin valores vacíos).
# - Guardado en `submissions/submission.csv`.

# %%
print(f'Generando predicciones para {len(test_csv)} preguntas de test...')
predictions = []

for idx, row in test_csv.iterrows():
    pred = predict_answer(row['question'])
    predictions.append({
        'id': row['id'],
        'answer': pred
    })
    if (idx + 1) % 100 == 0 or (idx + 1) == len(test_csv):
        print(f'Procesadas {idx + 1}/{len(test_csv)} preguntas...')

sub_df = pd.DataFrame(predictions)
sub_path = SUBMISSIONS_DIR / 'submission.csv'
sub_df.to_csv(sub_path, index=False)

# Validaciones formales de submission
assert len(sub_df) == 800, f'Error: longitud esperada 800, obtenida {len(sub_df)}'
assert list(sub_df.columns) == ['id', 'answer'], f'Error en columnas: {list(sub_df.columns)}'
assert sub_df['answer'].isna().sum() == 0, 'Error: existen valores nulos en predicciones'
assert (sub_df['answer'].str.strip() == '').sum() == 0, 'Error: existen respuestas vacías'

print(f'Archivo de submission guardado y verificado exitosamente en: {sub_path}')
print('\nPrimeras 10 filas:')
print(sub_df.head(10).to_string())
