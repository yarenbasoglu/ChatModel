import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from threadpoolctl import threadpool_limits

from chatmodel.data import load_splits, sha256_file
from chatmodel.evaluation import benchmark, classification_metrics, hardware_info
from chatmodel.models.baseline import BaselineModel
from chatmodel.models.neural import NeuralModel
from chatmodel.models.rules import EditDistanceModel
from chatmodel.models.bert import BertModel, artifact_digest, artifact_files, probabilities
from chatmodel.models.quantized_bert import QuantizedBertModel
from chatmodel.paths import QUANTIZED_BERT_MANIFEST
from chatmodel.paths import ARTIFACT, BERT_DIRECTORY, BERT_MANIFEST, COMPARISON_REPORT, MAX_MESSAGE_LENGTH, NEURAL_ARTIFACT, PROCESSED, RULE_CONFIG


def evaluate(output: Path, bert_device: str = "auto", reuse_report: Path | None = None) -> dict:
    splits, manifest = load_splits(PROCESSED)
    testing = splits["test"]
    eligible = testing.loc[testing.comment_text.str.len() <= MAX_MESSAGE_LENGTH]
    samples = eligible.sample(n=min(300, len(eligible)), random_state=42).comment_text.tolist()
    entries = [
        (BaselineModel.load(ARTIFACT), ARTIFACT, "trained"),
        (NeuralModel.load(NEURAL_ARTIFACT), NEURAL_ARTIFACT, "trained"),
        (EditDistanceModel.load(RULE_CONFIG), RULE_CONFIG, "rule_based"),
    ]
    if BERT_MANIFEST.is_file():
        entries.append((BertModel.load(BERT_DIRECTORY), BERT_DIRECTORY, "trained"))
    reused = {}
    if reuse_report:
        previous = json.loads(reuse_report.read_text(encoding="utf-8"))
        if previous["dataset"] != manifest:
            raise ValueError("Saved comparison uses a different dataset.")
        reused = {entry["model"]: entry for entry in previous["models"]}
    if QUANTIZED_BERT_MANIFEST.is_file():
        entries.append((QuantizedBertModel.load(QUANTIZED_BERT_MANIFEST), QUANTIZED_BERT_MANIFEST.parent, "quantized"))
    results = []
    with threadpool_limits(limits=1):
        for model, artifact, status in entries:
            print(f"Evaluating {model.name} on {len(testing):,} held-out messages...", flush=True)
            threshold = 0.5
            if status != "rule_based":
                if model.metadata["test_sha256"] != manifest["splits"]["test"]["sha256"]:
                    raise ValueError("Model test checksum does not match the shared test set.")
                if model.metadata["source_sha256"] != manifest["source_sha256"]:
                    raise ValueError("Model source data checksum mismatch.")
                threshold = model.threshold
            size_bytes = sum(filename.stat().st_size for filename in artifact_files(artifact)) if artifact.is_dir() else artifact.stat().st_size
            digest = artifact_digest(artifact) if artifact.is_dir() else sha256_file(artifact)
            if digest == reused.get(model.name, {}).get("artifact_sha256"):
                results.append(reused[model.name])
                continue
            scoring_device = "cpu"
            cpu_parity = None
            if isinstance(model, BertModel):
                import torch
                import numpy as np

                scoring_device = bert_device
                if scoring_device == "auto":
                    scoring_device = "mps" if torch.backends.mps.is_available() else "cpu"
                model.model.to(scoring_device)
                scores = probabilities(model.model, model.tokenizer, testing.comment_text.tolist(), scoring_device,
                                       model.max_length, batch_size=8,
                                       progress=lambda done, total: print(f"BERT test {done}/{total}", flush=True))
                model.model.to("cpu")
                cpu_scores = model.batch_scores(testing.comment_text.iloc[:32].tolist())
                difference = float(np.max(np.abs(cpu_scores - scores[:32])))
                if difference > 0.001:
                    raise ValueError(f"CPU/{scoring_device} probability discrepancy: {difference}")
                cpu_parity = {"samples": 32, "max_probability_difference": difference,
                              "matching_decisions": int(np.sum((cpu_scores >= threshold) == (scores[:32] >= threshold)))}
            elif isinstance(model, BaselineModel):
                scores = model.pipeline.predict_proba(testing.comment_text)[:, 1]
            else:
                scores = model.batch_scores(testing.comment_text.tolist())
            result = {"model": model.name, "status": status, "metric_device": scoring_device,
                      "test_metrics": classification_metrics(testing.label, scores, threshold),
                      "threshold": threshold, "artifact_size_mb": size_bytes / 1_000_000,
                      "artifact_sha256": digest, "latency": benchmark(model, samples)}
            if cpu_parity is not None:
                result["cpu_parity_check"] = cpu_parity
            result["latency"]["scope"] = "Single-message CPU prediction including validation, regex and selected model. Excludes loading/network/UI. BLAS limited to one thread for all models."
            if isinstance(model, QuantizedBertModel):
                result["latency"]["onnxruntime_intra_op_threads"] = model.cpu_threads
                result["latency"]["scope"] = "Single-message INT8 ONNX Runtime CPU prediction, including validation, regex and tokenization. Excludes loading/network/UI."
            results.append(result)
            print(json.dumps({"model": model.name, "macro_f1": result["test_metrics"]["macro_f1"], "mean_ms": result["latency"]["mean_ms"]}), flush=True)
    report = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": manifest,
              "hardware": hardware_info(), "models": results,
              "rule_note": "Rule size measures its JSON lexicon/config only; Python code and shared dependencies excluded. Rule ROC-AUC uses binary decisions, not calibrated probabilities.",
              "neural_note": "Inference bundle includes all trained Word2Vec vectors, vocabulary, scaler and network weights.",
              "bert_note": "BERT trained on a 12000-row stratified training subset, last 2 encoder layers fine-tuned. Metric device recorded per row; all latency measurements and UI predictions run on CPU. BERT size includes full FP32 weights and tokenizer."}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate available models on the exact shared test set.")
    parser.add_argument("--output", type=Path, default=COMPARISON_REPORT)
    parser.add_argument("--bert-device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--reuse-report", type=Path, help="Reuse metrics only when dataset and artifact hashes match.")
    arguments = parser.parse_args()
    evaluate(arguments.output, arguments.bert_device, arguments.reuse_report)


if __name__ == "__main__":
    main()
