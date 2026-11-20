import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from chatmodel.preprocessing import normalize_text

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def sha256_file(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw(frame: pd.DataFrame) -> None:
    missing = set(["id", "comment_text", *LABEL_COLUMNS]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Jigsaw columns: {sorted(missing)}")
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Jigsaw data must have nonempty, unique IDs.")
    if not frame[LABEL_COLUMNS].isin([0, 1]).all().all():
        raise ValueError("All six labels must be binary 0/1 without missing values.")


def make_splits(frame: pd.DataFrame, seed: int = 42) -> tuple[dict, dict]:
    validate_raw(frame)
    clean = frame[["id", "comment_text"]].copy()
    clean["label"] = frame[LABEL_COLUMNS].max(axis=1).astype(int)
    clean["normalized_text"] = clean["comment_text"].fillna("").map(normalize_text)
    valid = clean["normalized_text"].str.contains(r"[a-z]", regex=True)
    blank_count = int((~valid).sum())
    clean = clean.loc[valid].copy()
    conflicts = clean.groupby("normalized_text")["label"].nunique()
    conflict_texts = conflicts[conflicts > 1].index
    conflict_count = int(clean["normalized_text"].isin(conflict_texts).sum())
    clean = clean.loc[~clean["normalized_text"].isin(conflict_texts)]
    before_dedup = len(clean)
    clean = clean.drop_duplicates("normalized_text").reset_index(drop=True)
    if clean["label"].nunique() != 2 or clean["label"].value_counts().min() < 20:
        raise ValueError("At least 20 distinct usable examples per class are required.")
    train, holdout = train_test_split(clean, test_size=0.2, stratify=clean["label"], random_state=seed)
    validation, test = train_test_split(holdout, test_size=0.5, stratify=holdout["label"], random_state=seed)
    splits = {"train": train, "validation": validation, "test": test}
    summary = {
        "seed": seed,
        "raw_rows": len(frame),
        "removed_empty_rows": blank_count,
        "removed_conflicting_rows": conflict_count,
        "removed_duplicate_rows": before_dedup - len(clean),
        "usable_rows": len(clean),
        "label_definition": "Offensive = any of the six Jigsaw labels is 1",
        "splits": {
            name: {"rows": len(part), "clean": int((part.label == 0).sum()), "offensive": int((part.label == 1).sum())}
            for name, part in splits.items()
        },
    }
    return splits, summary


def prepare_dataset(source: Path, destination: Path, seed: int = 42) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Data not found: {source}. Run python -m scripts.download_data or provide Kaggle train.csv.")
    splits, summary = make_splits(pd.read_csv(source, dtype={"id": str}), seed)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "manifest.json").exists():
        raise FileExistsError("Prepared splits already exist. Use a new output directory to preserve the shared test set.")
    summary["source_sha256"] = sha256_file(source)
    for name, part in splits.items():
        output = destination / f"{name}.csv"
        part.to_csv(output, index=False)
        summary["splits"][name]["sha256"] = sha256_file(output)
    (destination / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_splits(directory: Path) -> tuple[dict, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    splits = {}
    for name in ("train", "validation", "test"):
        filename = directory / f"{name}.csv"
        if sha256_file(filename) != manifest["splits"][name]["sha256"]:
            raise ValueError(f"{name} split checksum mismatch. Refusing inconsistent evaluation data.")
        splits[name] = pd.read_csv(filename, dtype={"id": str})
    return splits, manifest
