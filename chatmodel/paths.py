from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "train.csv"
PROCESSED = ROOT / "data" / "processed"
ARTIFACT = ROOT / "artifacts" / "baseline.joblib"
REPORT = ROOT / "reports" / "baseline_metrics.json"
NEURAL_ARTIFACT = ROOT / "artifacts" / "neural.joblib"
NEURAL_REPORT = ROOT / "reports" / "neural_training.json"
RULE_CONFIG = ROOT / "chatmodel" / "models" / "rules_lexicon.json"
COMPARISON_REPORT = ROOT / "reports" / "comparison.json"
BERT_DIRECTORY = ROOT / "artifacts" / "bert"
BERT_MANIFEST = BERT_DIRECTORY / "training.json"
BERT_REPORT = ROOT / "reports" / "bert_training.json"
QUANTIZED_BERT_MANIFEST = ROOT / "artifacts" / "bert_int8" / "conversion.json"
MAX_MESSAGE_LENGTH = 5000
