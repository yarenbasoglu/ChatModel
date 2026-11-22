import json
from pathlib import Path
from time import perf_counter

import numpy as np

from chatmodel.models.base import Prediction
from chatmodel.preprocessing import normalize_text
from chatmodel.validation import validate_message

BASE_MODEL = "google-bert/bert-base-multilingual-cased"
BASE_REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"


def artifact_files(directory: Path) -> list[Path]:
    return sorted(filename for filename in directory.iterdir() if filename.is_file())


def artifact_digest(directory: Path) -> str:
    import hashlib
    from chatmodel.data import sha256_file

    digest = hashlib.sha256()
    for filename in artifact_files(directory):
        digest.update(f"{filename.name}:{sha256_file(filename)}\n".encode())
    return digest.hexdigest()


def probabilities(model, tokenizer, texts, device: str, max_length: int, batch_size: int = 16, progress=None):
    import torch

    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [normalize_text(text) for text in texts[start:start + batch_size]]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("BERT produced non-finite probabilities.")
            outputs.append(scores)
            if progress is not None and (start // batch_size) % 100 == 0:
                progress(min(start + batch_size, len(texts)), len(texts))
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


class BertModel:
    name = "Multilingual BERT"

    def __init__(self, directory: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        manifest = json.loads((directory / "training.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("status") != "trained":
            raise ValueError("BERT training is incomplete or the model file is invalid.")
        self.metadata = manifest["metadata"]
        self.threshold = manifest["threshold"]
        if not 0 < self.threshold < 1:
            raise ValueError("Invalid BERT decision threshold.")
        self.max_length = self.metadata["max_length"]
        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, fix_mistral_regex=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True, attn_implementation="eager")
        self.model.to("cpu").eval()

    @classmethod
    def load(cls, filename: Path):
        return cls(filename.parent if filename.is_file() else filename)

    def batch_scores(self, texts, progress=None):
        return probabilities(self.model, self.tokenizer, texts, "cpu", self.max_length, progress=progress)

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
