import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import sklearn
from sklearn.exceptions import ConvergenceWarning

from chatmodel.data import load_splits, sha256_file
from chatmodel.evaluation import benchmark, choose_threshold, classification_metrics, hardware_info
from chatmodel.models.baseline import BaselineModel, build_pipeline
from chatmodel.paths import ARTIFACT, MAX_MESSAGE_LENGTH, PROCESSED, REPORT


def train(data_directory: Path, artifact: Path, report_path: Path) -> dict:
    if artifact.exists() or report_path.exists():
        raise FileExistsError("Existing model/report preserved. Choose new --artifact and --report paths for another run.")
    splits, manifest = load_splits(data_directory)
    training = splits["train"]
    validation = splits["validation"]
    testing = splits["test"]
    pipeline = build_pipeline()
    started = perf_counter()
    print(f"Fitting on {len(training):,} real training examples...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(training.comment_text, training.label)
    training_seconds = perf_counter() - started
    print("Selecting threshold on validation only...", flush=True)
    threshold, validation_f1 = choose_threshold(validation.label, pipeline.predict_proba(validation.comment_text)[:, 1])
    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__, "training_seconds": training_seconds,
        "training_rows": len(training), "source_sha256": manifest["source_sha256"],
        "test_sha256": manifest["splits"]["test"]["sha256"], "language": "English",
        "normalization_version": 1,
    }
    model = BaselineModel(pipeline, threshold, metadata)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "status": "trained", "pipeline": pipeline, "threshold": threshold, "metadata": metadata}, artifact, compress=3)
    print("Evaluating untouched test split and measuring CPU latency...", flush=True)
    metrics = classification_metrics(testing.label, pipeline.predict_proba(testing.comment_text)[:, 1], threshold)
    eligible = testing.loc[testing.comment_text.str.len() <= MAX_MESSAGE_LENGTH]
    samples = eligible.sample(n=min(300, len(eligible)), random_state=42).comment_text.tolist()
    report = {
        "model": model.name, "status": "trained", "metadata": metadata,
        "dataset": manifest, "threshold": threshold, "validation_macro_f1": validation_f1,
        "test_metrics": metrics, "artifact_size_mb": artifact.stat().st_size / 1_000_000,
        "artifact_sha256": sha256_file(artifact), "latency": benchmark(model, samples),
        "hardware": hardware_info(),
        "probability_note": "Logistic regression scores are not separately calibrated; confidence is not a correctness guarantee.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the real TF-IDF baseline.")
    parser.add_argument("--data-dir", type=Path, default=PROCESSED)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--report", type=Path, default=REPORT)
    arguments = parser.parse_args()
    train(arguments.data_dir, arguments.artifact, arguments.report)


if __name__ == "__main__":
    main()
