import json
from pathlib import Path
from time import perf_counter

import numpy as np
from rapidfuzz.distance import Levenshtein

from chatmodel.models.base import Prediction
from chatmodel.preprocessing import normalize_text
from chatmodel.validation import validate_message


class EditDistanceModel:
    name = "Edit-distance sözlük kuralı"

    def __init__(self, configuration: dict):
        if configuration.get("schema_version") != 1 or configuration.get("status") != "configured":
            raise ValueError("Geçersiz sözlük yapılandırması.")
        self.terms = frozenset(configuration["terms"])
        if not self.terms or any(normalize_text(term) != term or " " in term for term in self.terms):
            raise ValueError("Sözlük normalize edilmiş tek sözcükler içermeli.")
        self.max_distance = configuration["max_distance"]
        self.minimum_fuzzy_length = configuration["minimum_fuzzy_length"]
        if self.max_distance != 1 or self.minimum_fuzzy_length < 5:
            raise ValueError("Bu sürüm en az 5 harfte en fazla 1 düzenlemeyi destekler.")
        self.by_length = {}
        for term in sorted(self.terms):
            self.by_length.setdefault(len(term), []).append(term)

    @classmethod
    def load(cls, filename: Path):
        return cls(json.loads(filename.read_text(encoding="utf-8")))

    def match_score(self, normalized: str) -> float:
        tokens = set(normalized.split())
        if tokens & self.terms:
            return 1.0
        best_score = 0.0
        for token in tokens:
            if len(token) < self.minimum_fuzzy_length:
                continue
            for length in range(max(self.minimum_fuzzy_length, len(token) - 1), len(token) + 2):
                for term in self.by_length.get(length, []):
                    distance = Levenshtein.distance(token, term, score_cutoff=self.max_distance)
                    if distance <= self.max_distance:
                        best_score = max(best_score, 1 - distance / max(len(token), len(term)))
        return best_score

    def batch_scores(self, texts) -> np.ndarray:
        return np.array([float(self.match_score(normalize_text(text)) > 0) for text in texts])

    def predict(self, text: str) -> Prediction:
        started = perf_counter()
        normalized = validate_message(text)
        score = self.match_score(normalized)
        return Prediction(
            label="Offensive" if score > 0 else "Clean", confidence=score,
            offensive_probability=None, latency_ms=(perf_counter() - started) * 1000,
            normalized_text=normalized, model_name=self.name, score_kind="lexicon_similarity",
        )
