import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import softmax

from chatmodel.models.base import Prediction
from chatmodel.preprocessing import normalize_text
from chatmodel.validation import validate_message


class QuantizedBertModel:
    name = "Multilingual BERT INT8"

    def __init__(self, directory: Path, cpu_threads: int | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        manifest = json.loads((directory / "conversion.json").read_text(encoding="utf-8"))
        if manifest["format"] != "onnx-dynamic-int8":
            raise ValueError("Invalid INT8 model format.")
        self.metadata = manifest["metadata"]
        self.max_length = self.metadata["max_length"]
        self.threshold = manifest["threshold"]
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, fix_mistral_regex=False)
        options = ort.SessionOptions()
        self.cpu_threads = cpu_threads if cpu_threads is not None else manifest.get("cpu_threads", 1)
        options.intra_op_num_threads = self.cpu_threads
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"],
        )
        self.input_names = {value.name for value in self.session.get_inputs()}

    @classmethod
    def load(cls, manifest: Path):
        return cls(manifest.parent)

    def _scores(self, encoded):
        inputs = {key: value for key, value in encoded.items() if key in self.input_names}
        logits = self.session.run(["logits"], inputs)[0]
        if not np.isfinite(logits).all():
            raise ValueError("INT8 model returned non-finite logits.")
        return softmax(logits, axis=-1)[:, 1]

    def batch_scores(self, texts, batch_size=32):
        if not texts:
            return np.empty(0, dtype=np.float32)
        encoded = self.tokenizer([normalize_text(text) for text in texts], truncation=True, max_length=self.max_length)
        order = sorted(range(len(texts)), key=lambda index: len(encoded["input_ids"][index]))
        scores = np.empty(len(texts), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            indices = order[start:start + batch_size]
            batch = [{key: values[index] for key, values in encoded.items()} for index in indices]
            scores[indices] = self._scores(self.tokenizer.pad(batch, padding=True, return_tensors="np"))
            if start and start % (batch_size * 100) == 0:
                print(f"INT8 test {start}/{len(texts)}", flush=True)
        return scores

    def predict(self, text: str) -> Prediction:
        started = perf_counter()
        normalized = validate_message(text)
        encoded = self.tokenizer([normalized], truncation=True, max_length=self.max_length, return_tensors="np")
        probability = float(self._scores(encoded)[0])
        offensive = probability >= self.threshold
        return Prediction(
            label="Offensive" if offensive else "Clean",
            confidence=probability if offensive else 1 - probability,
            offensive_probability=probability, latency_ms=(perf_counter() - started) * 1000,
            normalized_text=normalized, model_name=self.name,
        )
