import json
from pathlib import Path
import re
from collections import Counter
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
from scripts.normalizer import normalize_answer

DATA_DIR = Path("data")
with open(DATA_DIR / "train.jsonl", encoding="utf-8") as f:
    train_jsonl = [json.loads(line) for line in f if line.strip()]

KB_DIR = DATA_DIR / "base_conocimiento"
doc_text_map = {p.parent.name: p.read_text(encoding="utf-8") for p in sorted(KB_DIR.glob("*/doc.md"))}

# Cargar LLM
dec_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    local_files_only=True,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
dec_model = PeftModel.from_pretrained(base_model, "models/decoder")
dec_model.eval()

PROMPT_TEMPLATE = """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query concisely with a short canonical phrase, role, value with unit, or protocol ID.
Query: {query}
Answer: """

def predict(context, query):
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)
    inputs = dec_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = dec_model.generate(**inputs, max_new_tokens=25, do_sample=False, pad_token_id=dec_tokenizer.eos_token_id, eos_token_id=dec_tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = dec_tokenizer.decode(gen, skip_special_tokens=True).strip().split("\n")[0].strip()
    return normalize_answer(ans)

# Probar en 100 preguntas de train con contexto PERFECTO (sus support_doc_ids)
hits = 0
sample = train_jsonl[:100]
print(f"Probando decoder sobre {len(sample)} preguntas de train con soporte documental real...")
for x in sample:
    ctx = "\n\n".join([doc_text_map[d] for d in x["support_doc_ids"]])
    pred = predict(ctx, x["question"])
    gold = normalize_answer(x["answer"])
    if pred == gold:
        hits += 1
    else:
        print(f"FAIL: Q='{x['question']}'")
        print(f"      PRED='{pred}' vs GOLD='{gold}'")

print(f"\nExact Match con contexto perfecto: {hits}/{len(sample)} ({hits/len(sample)*100:.2f}%)")
