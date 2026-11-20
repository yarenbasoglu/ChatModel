import re

from chatmodel.paths import MAX_MESSAGE_LENGTH
from chatmodel.preprocessing import normalize_text


def validate_message(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Lütfen boş olmayan bir mesaj yazın.")
    if len(text) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Mesaj en fazla {MAX_MESSAGE_LENGTH} karakter olabilir.")
    normalized = normalize_text(text)
    if not re.search(r"[a-z]", normalized):
        raise ValueError("Analiz için İngilizce sözcük içeren bir mesaj yazın.")
    return normalized
