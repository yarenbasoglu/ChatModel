from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn

from chatmodel.models.base import Prediction
from chatmodel.preprocessing import normalize_text
from chatmodel.validation import validate_message


def average_embeddings(texts, vocabulary: dict, vectors: np.ndarray) -> np.ndarray:
    features = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    for row, text in enumerate(texts):
        indices = [vocabulary[token] for token in normalize_text(text).split() if token in vocabulary]
        if indices:
            features[row] = vectors[indices].mean(axis=0)
    return features


class NeuralModel:
    name = "Word2Vec + Neural Network"

    def __init__(self, bundle: dict):
        if bundle.get("schema_version") != 1 or bundle.get("status") != "trained":
            raise ValueError("Geçersiz veya eğitilmemiş sinir ağı dosyası.")
        self.metadata = bundle["metadata"]
        if self.metadata["sklearn_version"] != sklearn.__version__:
            raise ValueError("Sinir ağı scikit-learn sürümü uyuşmuyor. requirements-lock.txt kullanın.")
        self.vocabulary = bundle["vocabulary"]
        self.vectors = bundle["vectors"]
        self.scaler = bundle["scaler"]
        self.classifier = bundle["classifier"]
        self.threshold = bundle["threshold"]
        if not 0 < self.threshold < 1:
            raise ValueError("Geçersiz sinir ağı karar eşiği.")

    @classmethod
    def load(cls, filename: Path):
        return cls(joblib.load(filename))

    def batch_scores(self, texts) -> np.ndarray:
        features = average_embeddings(texts, self.vocabulary, self.vectors)
        return self.classifier.predict_proba(self.scaler.transform(features))[:, 1]

    def predict(self, text: str) -> Prediction:
        started = perf_counter()
        normalized = validate_message(text)
        probability = float(self.batch_scores([text])[0])
        offensive = probability >= self.threshold
        return Prediction(
            label="Offensive" if offensive else "Clean",
            confidence=probability if offensive else 1 - probability,
            offensive_probability=probability, latency_ms=(perf_counter() - started) * 1000,
            normalized_text=normalized, model_name=self.name,
        )
