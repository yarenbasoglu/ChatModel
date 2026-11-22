from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from chatmodel.models.base import Prediction
from chatmodel.preprocessing import normalize_text
from chatmodel.validation import validate_message


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_text, lowercase=False, ngram_range=(1, 2),
            min_df=2, max_df=0.98, max_features=120000, sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b", dtype=np.float64,
        )),
        ("classifier", LogisticRegression(
            solver="liblinear", C=4.0, class_weight="balanced", max_iter=1000, random_state=42,
        )),
    ])


class BaselineModel:
    name = "TF-IDF + Logistic Regression"

    def __init__(self, pipeline: Pipeline, threshold: float, metadata: dict):
        if not 0 < threshold < 1:
            raise ValueError("Threshold must be between 0 and 1.")
        self.pipeline = pipeline
        self.threshold = threshold
        self.metadata = metadata

    @classmethod
    def load(cls, filename: Path):
        if not filename.is_file():
            raise FileNotFoundError("Trained model not found. Run python -m scripts.train_baseline first.")
        bundle = joblib.load(filename)
        if bundle.get("schema_version") != 1 or bundle.get("status") != "trained":
            raise ValueError("Invalid or untrained model file.")
        if bundle["metadata"]["sklearn_version"] != sklearn.__version__:
            raise ValueError("The model requires a different scikit-learn version. Install the dependencies in requirements-lock.txt or retrain the model.")
        return cls(bundle["pipeline"], bundle["threshold"], bundle["metadata"])

    def predict(self, text: str) -> Prediction:
        started = perf_counter()
        normalized = validate_message(text)
        probability = float(self.pipeline.predict_proba([text])[0, 1])
        offensive = probability >= self.threshold
        return Prediction(
            label="Offensive" if offensive else "Clean",
            confidence=probability if offensive else 1 - probability,
            offensive_probability=probability,
            latency_ms=(perf_counter() - started) * 1000,
            normalized_text=normalized,
            model_name=self.name,
        )
