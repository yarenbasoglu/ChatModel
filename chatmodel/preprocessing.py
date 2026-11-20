import re
import unicodedata

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.IGNORECASE)
TOKEN_PATTERN = re.compile(r"[a-z0-9@$]+(?:'[a-z]+)?")
REPEATED_PATTERN = re.compile(r"([a-z])\1{2,}")
LEET_TABLE = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
TURKISH_TABLE = str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ü": "u", "ö": "o", "ç": "c"})


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Message must be a string.")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("\u0307", "").translate(TURKISH_TABLE)
    normalized = re.sub(r"[\u200b-\u200d\ufeff]", "", normalized)
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = EMAIL_PATTERN.sub(" ", normalized)
    tokens = TOKEN_PATTERN.findall(normalized)
    words = []
    for token in tokens:
        if re.search(r"[a-z]", token):
            token = token.translate(LEET_TABLE)
            words.append(REPEATED_PATTERN.sub(r"\1\1", token))
        elif token.isdigit():
            words.append(token)
    return " ".join(words)
