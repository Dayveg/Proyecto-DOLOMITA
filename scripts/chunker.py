import re
from pathlib import Path
from typing import List, Dict, Any

def parse_markdown_document(doc_path: Path) -> List[Dict[str, Any]]:
    """
    Parsea un documento markdown de la base de conocimiento en chunks estructurados.
    Preserva el doc_id, el título del documento (H1), el subtítulo de la sección (H2) y tablas completas.
    """
    doc_id = doc_path.parent.name
    text = doc_path.read_text(encoding="utf-8").strip()
    
    # Extraer H1
    h1_match = re.search(r"^#\s+(.+)$", text, re.M)
    doc_title = h1_match.group(1).strip() if h1_match else doc_id
    
    # Dividir por secciones H2 (## )
    sections = re.split(r"(^##\s+.+$)", text, flags=re.M)
    
    chunks = []
    chunk_idx = 0
    
    # Si hay contenido previo a cualquier H2
    preamble = sections[0].strip()
    if preamble:
        chunk_text = f"Document: {doc_id} - {doc_title}\n{preamble}"
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_idx,
            "section": "Overview",
            "text": chunk_text
        })
        chunk_idx += 1
        
    for i in range(1, len(sections), 2):
        sec_header = sections[i].strip()
        sec_title = re.sub(r"^##\s+", "", sec_header).strip()
        sec_content = sections[i+1].strip() if i+1 < len(sections) else ""
        
        # Limpieza de separadores markdown '---'
        sec_content = re.sub(r"\n---\n", "\n", sec_content)
        
        full_chunk_text = f"Document: {doc_id} - {doc_title} | Section: {sec_title}\n{sec_content}"
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_idx,
            "section": sec_title,
            "text": full_chunk_text.strip()
        })
        chunk_idx += 1
        
    return chunks

def get_all_chunks(kb_dir: Path) -> List[Dict[str, Any]]:
    all_chunks = []
    global_chunk_id = 0
    for doc_file in sorted(kb_dir.glob("*/doc.md")):
        doc_chunks = parse_markdown_document(doc_file)
        for c in doc_chunks:
            c["global_chunk_id"] = global_chunk_id
            global_chunk_id += 1
            all_chunks.append(c)
    return all_chunks

if __name__ == "__main__":
    kb = Path("data/base_conocimiento")
    chunks = get_all_chunks(kb)
    print(f"Total chunks creados: {len(chunks)} a partir de 300 documentos.")
    print("Ejemplo primer chunk:")
    print(chunks[0])
