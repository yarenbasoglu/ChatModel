from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    offensive_probability: float | None
    latency_ms: float
    normalized_text: str
    model_name: str
    score_kind: str = "probability"


class ModerationModel(Protocol):
    def predict(self, text: str) -> Prediction: ...
