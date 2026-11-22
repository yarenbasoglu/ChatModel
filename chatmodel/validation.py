import re

from chatmodel.paths import MAX_MESSAGE_LENGTH
from chatmodel.preprocessing import normalize_text


def validate_message(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Enter a message.")
    if len(text) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Messages can contain up to {MAX_MESSAGE_LENGTH} characters.")
    normalized = normalize_text(text)
    if not re.search(r"[a-z]", normalized):
        raise ValueError("Enter a message containing at least one English word.")
    return normalized
