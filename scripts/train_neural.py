import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import gensim
import joblib
import numpy as np
import sklearn
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from chatmodel.data import load_splits, sha256_file
from chatmodel.evaluation import choose_threshold, hardware_info
from chatmodel.models.neural import average_embeddings
from chatmodel.paths import NEURAL_ARTIFACT, NEURAL_REPORT, PROCESSED
from chatmodel.preprocessing import normalize_text
from scripts.fix_gensim_blas import verify_backend


def stable_hash(word: str) -> int:
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:4], "little")


class Word2VecProgress(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Word2Vec epoch {self.epoch}/{model.epochs} complete", flush=True)


def write_training_report(bundle: dict, artifact: Path, report_path: Path) -> dict:
    history = bundle["training_history"]
    best_epoch = bundle["metadata"]["network"]["best_epoch"]
    best_record = next(record for record in history if record["epoch"] == best_epoch)
    report = {"model": "Word2Vec + Neural Network", "status": "trained", "metadata": bundle["metadata"],
              "threshold": bundle["threshold"], "validation_macro_f1": best_record["validation_macro_f1"],
              "history": history, "artifact_sha256": sha256_file(artifact),
              "artifact_size_mb": artifact.stat().st_size / 1_000_000,
              "hardware": hardware_info(), "test_evaluation": "Run python -m scripts.evaluate_models"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return report


def train(data_directory: Path, artifact: Path, report_path: Path, epochs: int = 30) -> dict:
    verify_backend()
    if epochs < 1:
        raise ValueError("Epoch count must be positive.")
    if artifact.exists() or report_path.exists():
        raise FileExistsError("Model/report already exists. Select new --artifact and --report paths.")
    splits, manifest = load_splits(data_directory)
    training = splits["train"]
    validation = splits["validation"]
    started = perf_counter()
    sentences = [normalize_text(text).split() for text in training.comment_text]
    print(f"Training own Word2Vec on {len(sentences):,} training messages...", flush=True)
    embedding = Word2Vec(
        sentences=sentences, vector_size=100, window=5, min_count=3,
        sg=1, negative=5, epochs=5, workers=1, seed=42, hashfxn=stable_hash,
        callbacks=[Word2VecProgress()],
    )
    word2vec_seconds = perf_counter() - started
    vocabulary = dict(embedding.wv.key_to_index)
    vectors = embedding.wv.vectors.copy()
    if not np.isfinite(vectors).all():
        raise ValueError("Word2Vec produced non-finite vectors.")
    print(f"Embedding vocabulary: {len(vocabulary):,}; building averaged features...", flush=True)
    features = average_embeddings(training.comment_text.tolist(), vocabulary, vectors)
    validation_features = average_embeddings(validation.comment_text.tolist(), vocabulary, vectors)
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    validation_features = scaler.transform(validation_features)
    weights = compute_sample_weight("balanced", training.label)
    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 32), activation="relu", solver="sgd",
        learning_rate_init=0.01, learning_rate="constant", momentum=0.9,
        alpha=0.0001, batch_size=256, shuffle=True, random_state=42,
        early_stopping=False,
    )
    history = []
    best_score = -1.0
    stale_epochs = 0
    with threadpool_limits(limits=1):
        for epoch in range(1, epochs + 1):
            classifier.partial_fit(features, training.label, classes=np.array([0, 1]), sample_weight=weights)
            probabilities = classifier.predict_proba(validation_features)[:, 1]
            threshold, score = choose_threshold(validation.label, probabilities)
            record = {"epoch": epoch, "training_loss": float(classifier.loss_), "validation_macro_f1": score, "threshold": threshold}
            history.append(record)
            print(json.dumps(record), flush=True)
            if score > best_score + 0.0001:
                best_score = score
                best_classifier = copy.deepcopy(classifier)
                best_threshold = threshold
                best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 5:
                break
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(training), "source_sha256": manifest["source_sha256"],
        "test_sha256": manifest["splits"]["test"]["sha256"],
        "train_sha256": manifest["splits"]["train"]["sha256"],
        "sklearn_version": sklearn.__version__, "gensim_version": gensim.__version__,
        "gensim_blas_fix": "issue #3665; sdot_ptr and dsdot_ptr noexcept; regression verified before training",
        "language": "English", "normalization_version": 1, "seed": 42,
        "word2vec": {"architecture": "skip-gram", "vector_size": 100, "window": 5, "min_count": 3,
                     "epochs": 5, "negative": 5, "workers": 1, "vocabulary_size": len(vocabulary),
                     "training_seconds": word2vec_seconds, "hash_function": "SHA256 first 32 bits"},
        "network": {"layers": [100, 128, 32, 1], "hidden_activation": "relu", "output_activation": "logistic",
                    "optimizer": "SGD with momentum 0.9 and backpropagation", "learning_rate": 0.01,
                    "batch_size": 256, "class_weight": "balanced sample weights from train",
                    "max_epochs": epochs, "best_epoch": best_epoch, "early_stopping_patience": 5},
        "training_seconds": perf_counter() - started,
    }
    bundle = {"schema_version": 1, "status": "trained", "metadata": metadata,
              "vocabulary": vocabulary, "vectors": vectors, "scaler": scaler,
              "classifier": best_classifier, "threshold": best_threshold, "training_history": history}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(artifact)
    report = write_training_report(bundle, artifact, report_path)
    print(f"Saved real trained model: {artifact}; best validation Macro-F1={best_score:.4f}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Train own Word2Vec and SGD feedforward network on the fixed Jigsaw split.")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED)
    parser.add_argument("--artifact", type=Path, default=NEURAL_ARTIFACT)
    parser.add_argument("--report", type=Path, default=NEURAL_REPORT)
    parser.add_argument("--epochs", type=int, default=30)
    arguments = parser.parse_args()
    train(arguments.data_dir, arguments.artifact, arguments.report, arguments.epochs)


if __name__ == "__main__":
    main()
