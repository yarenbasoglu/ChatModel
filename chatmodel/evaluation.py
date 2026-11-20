import json
import platform
import subprocess
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score


def choose_threshold(labels, probabilities) -> tuple[float, float]:
    probabilities = np.asarray(probabilities)
    candidates = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(labels, probabilities >= threshold, average="macro", zero_division=0) for threshold in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def classification_metrics(labels, probabilities, threshold: float) -> dict:
    probabilities = np.asarray(probabilities)
    predicted = probabilities >= threshold
    return {
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix_labels": ["Clean", "Offensive"],
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(labels, predicted, labels=[0, 1], target_names=["Clean", "Offensive"], output_dict=True, zero_division=0),
    }


def hardware_info() -> dict:
    processor = platform.processor()
    if platform.system() == "Darwin":
        try:
            processor = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            try:
                details = subprocess.check_output(
                    ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
                    text=True, stderr=subprocess.DEVNULL, timeout=5,
                )
                hardware = json.loads(details)["SPHardwareDataType"][0]
                processor = hardware.get("chip_type") or hardware["cpu_type"]
            except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
                processor = f"{processor or platform.machine()} (CPU model query unavailable)"
    return {"os": platform.platform(), "cpu": processor, "machine": platform.machine(), "python": platform.python_version()}


def benchmark(model, texts: list[str], runs: int = 300) -> dict:
    if runs < 100 or not texts:
        raise ValueError("Benchmark needs at least 100 runs and nonempty messages.")
    for text in texts[:10]:
        model.predict(text)
    durations = []
    for index in range(runs):
        started = perf_counter()
        model.predict(texts[index % len(texts)])
        durations.append((perf_counter() - started) * 1000)
    return {
        "runs": runs, "warmup_runs": min(10, len(texts)), "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95)),
        "max_ms": float(np.max(durations)), "budget_ms": 50.0,
        "p95_within_budget": bool(np.percentile(durations, 95) <= 50),
        "scope": "Single-message CPU prediction including validation, regex, TF-IDF and classifier; excludes loading, network and UI rendering.",
        "message_characters": {"min": min(map(len, texts)), "max": max(map(len, texts))},
    }
