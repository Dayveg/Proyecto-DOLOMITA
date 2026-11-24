import re

def normalize_answer(text: str) -> str:
    """
    Normalización determinista según las reglas oficiales de la Sección 8.1:
    1. Conversión a minúsculas, excepto identificadores de protocolo (MC26-SEC-XXX) en MAYÚSCULAS.
    2. Recorte de espacios y colapso de múltiples espacios a uno solo.
    3. Eliminación de puntuación final (. , ; : ! ?) y de artículos iniciales (a, an, the).
    4. Números en dígitos; unidades canónicas con espacio (e.g., 40 cm, 120 km/h).
    5. Temperaturas como '<n> c'.
    6. Menos Unicode (−) a guion ASCII (-); guion bajo (_) a espacio.
    7. Booleanas como yes / no.
    """
    if text is None:
        return ""
    text = str(text)
    
    # 6. Menos Unicode, guiones largos y guiones bajos
    text = text.replace("\u2212", "-").replace("–", "-").replace("—", "-")
    text = text.replace("_", " ")
    
    # 2. Espacios iniciales/finales y colapso de espacios
    text = re.sub(r"\s+", " ", text).strip()
    
    # 5. Normalizar temperaturas: manejar °C, ℃, deg c, c
    text = text.replace("℃", " °C").replace("°c", " °C").replace("°C", " °C")
    text = re.sub(r"(-?\d+)\s*(?:°\s*c|deg\s*c|celsius)\b", r"\1 c", text, flags=re.IGNORECASE)
    text = re.sub(r"(-?\d+)\s*°\s*c\b", r"\1 c", text, flags=re.IGNORECASE)
    
    # Proteger identificadores de protocolo (MC26-SEC-XXX)
    protocols = re.findall(r"\bMC26-SEC-\d+\b", text, flags=re.IGNORECASE)
    proto_map = {}
    for i, p in enumerate(protocols):
        placeholder = f"__PROTO_{i}__"
        proto_map[placeholder] = p.upper()
        text = re.sub(re.escape(p), placeholder, text, count=1, flags=re.IGNORECASE)
        
    # 1. Pasar a minúsculas
    text = text.lower()
    
    # 3. Eliminar puntuación final
    text = re.sub(r"[.,;:!?]+$", "", text).strip()
    
    # 3. Eliminar artículos iniciales (a, an, the)
    text = re.sub(r"^(a|an|the)\s+", "", text).strip()
    
    # 4. Asegurar espacio en unidades canónicas comunes
    text = re.sub(r"(\d+)\s*(km/h|cm|m|minutes?|hours?|seconds?|c)\b", r"\1 \2", text)
    text = re.sub(r"\bminute\b", "minutes", text)
    text = re.sub(r"\bhour\b", "hours", text)
    text = re.sub(r"\bsecond\b", "seconds", text)
    
    # 7. Booleanas
    if text in ("true", "yes", "y"):
        text = "yes"
    elif text in ("false", "no", "n"):
        text = "no"
        
    # Restaurar protocolos en MAYÚSCULAS
    for placeholder, proto_upper in proto_map.items():
        text = text.replace(placeholder.lower(), proto_upper)
        
    return text.strip()

def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)

if __name__ == "__main__":
    test_cases = [
        ("The MC26-sec-009.", "MC26-SEC-009"),
        ("-22 °C", "-22 c"),
        ("-22℃", "-22 c"),
        ("-22c", "-22 c"),
        ("120km/h", "120 km/h"),
        ("Chief of Competition,", "chief of competition"),
        ("A diesel generator", "diesel generator"),
        ("venue_only", "venue only"),
        ("5 minute", "5 minutes"),
        ("Yes", "yes")
    ]
    for p, g in test_cases:
        norm_p = normalize_answer(p)
        norm_g = normalize_answer(g)
        assert norm_p == norm_g, f"Mismatch: {norm_p} vs {norm_g}"
    print("Normalizer tests passed successfully!")
