import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from chatmodel.data import load_splits
from chatmodel.evaluation import choose_threshold, hardware_info
from chatmodel.models.bert import BASE_MODEL, BASE_REVISION, probabilities
from chatmodel.paths import BERT_DIRECTORY, BERT_REPORT, PROCESSED, ROOT
from chatmodel.preprocessing import normalize_text


def select_subset(frame, limit: int):
    if limit <= 0 or limit >= len(frame):
        return frame.reset_index(drop=True)
    selected, _ = train_test_split(frame, train_size=limit, stratify=frame.label, random_state=42)
    return selected.reset_index(drop=True)


def freeze_encoder(model, trainable_layers: int = 2):
    if not 1 <= trainable_layers <= len(model.bert.encoder.layer):
        raise ValueError("Trainable layer count is outside the encoder range.")
    for parameter in model.bert.parameters():
        parameter.requires_grad = False
    for layer in model.bert.encoder.layer[-trainable_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.bert.pooler.parameters():
        parameter.requires_grad = True


def write_json(filename: Path, payload: dict):
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(filename)


def train(arguments):
    if (BERT_DIRECTORY / "training.json").exists():
        raise FileExistsError("Trained BERT already exists. Preserve it before starting another experiment.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = arguments.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Run with --device cpu or allow GPU access.")
    splits, manifest = load_splits(PROCESSED)
    training = select_subset(splits["train"], arguments.train_rows)
    monitor = select_subset(splits["validation"], 2000)
    validation = splits["validation"]
    base = ROOT / "data" / "raw" / "bert_base"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, fix_mistral_regex=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, local_files_only=True, attn_implementation="eager",
        id2label={0: "Clean", 1: "Offensive"}, label2id={"Clean": 0, "Offensive": 1},
    )
    freeze_encoder(model)
    parameter_counts = {"total": sum(parameter.numel() for parameter in model.parameters()),
                        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    model.to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-5, weight_decay=0.01)
    counts = np.bincount(training.label, minlength=2)
    class_weights = torch.tensor(len(training) / (2 * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    encoded = tokenizer([normalize_text(text) for text in training.comment_text], truncation=True, max_length=arguments.max_length)
    dataset = [{**{key: values[index] for key, values in encoded.items()}, "labels": int(label)} for index, label in enumerate(training.label)]
    run_directory = ROOT / "artifacts" / "bert_run"
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "last.pt"
    history = []
    best_score = -1.0
    starting_epoch = 0
    starting_batch = 0
    elapsed_before = 0.0
    configuration = {"train_rows": len(training), "batch_size": arguments.batch_size,
                     "max_length": arguments.max_length, "epochs": arguments.epochs,
                     "test_sha256": manifest["splits"]["test"]["sha256"],
                     "train_sha256": manifest["splits"]["train"]["sha256"]}
    if arguments.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["configuration"] != configuration:
            raise ValueError("Resume settings differ from saved checkpoint.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        best_score = checkpoint["best_score"]
        starting_epoch, starting_batch = checkpoint["epoch"], checkpoint["batch"]
        elapsed_before = checkpoint["elapsed_seconds"]
        torch.set_rng_state(checkpoint["cpu_rng"].cpu())
        if device == "mps" and checkpoint.get("mps_rng") is not None:
            torch.mps.set_rng_state(checkpoint["mps_rng"].cpu())
    elif checkpoint_path.exists():
        raise FileExistsError("A training checkpoint exists. Use --resume to continue it.")
    started = time.perf_counter()
    print(f"Fine-tuning {BASE_MODEL} on {len(training)} messages; device={device}; last 2 encoder layers trainable", flush=True)

    def save_checkpoint(epoch, batch):
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "batch": batch, "history": history, "best_score": best_score,
                    "cpu_rng": torch.get_rng_state(),
                    "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
                    "configuration": configuration, "elapsed_seconds": elapsed_before + time.perf_counter() - started}, temporary)
        temporary.replace(checkpoint_path)

    for epoch in range(starting_epoch, arguments.epochs):
        generator = torch.Generator().manual_seed(42 + epoch)
        loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True, generator=generator,
                            collate_fn=DataCollatorWithPadding(tokenizer), num_workers=0)
        model.train()
        model.bert.embeddings.eval()
        for layer in model.bert.encoder.layer[:-2]:
            layer.eval()
        losses = []
        for batch_index, batch in enumerate(loader):
            if epoch == starting_epoch and batch_index < starting_batch:
                continue
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite BERT training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if (batch_index + 1) % 25 == 0:
                progress = {"epoch": epoch + 1, "batch": batch_index + 1, "batches": len(loader),
                            "loss": float(np.mean(losses[-25:])), "elapsed_seconds": round(elapsed_before + time.perf_counter() - started)}
                print(json.dumps(progress), flush=True)
                write_json(run_directory / "progress.json", progress)
            if (batch_index + 1) % 250 == 0:
                save_checkpoint(epoch, batch_index + 1)
        scores = probabilities(model, tokenizer, monitor.comment_text.tolist(), device, arguments.max_length)
        threshold, score = choose_threshold(monitor.label, scores)
        record = {"epoch": epoch + 1, "monitor_rows": len(monitor), "validation_macro_f1": score,
                  "threshold": threshold, "mean_loss_since_resume": float(np.mean(losses)) if losses else None}
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            model.save_pretrained(run_directory / "best", safe_serialization=True)
            tokenizer.save_pretrained(run_directory / "best")
        save_checkpoint(epoch + 1, 0)
        starting_batch = 0
    del optimizer, model
    if device == "mps":
        torch.mps.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(run_directory / "best", local_files_only=True, attn_implementation="eager").to(device)
    print("Selecting final threshold on the entire validation split...", flush=True)
    scores = probabilities(model, tokenizer, validation.comment_text.tolist(), device, arguments.max_length,
                           progress=lambda done, total: print(f"Validation {done}/{total}", flush=True))
    threshold, validation_f1 = choose_threshold(validation.label, scores)
    selected_ids = training.id.tolist()
    metadata = {"trained_at_utc": datetime.now(timezone.utc).isoformat(), "base_model": BASE_MODEL,
                "base_revision": BASE_REVISION, "training_rows": len(training), "available_training_rows": len(splits["train"]),
                "training_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
                "source_sha256": manifest["source_sha256"], "test_sha256": manifest["splits"]["test"]["sha256"],
                "train_sha256": manifest["splits"]["train"]["sha256"], "language": "English",
                "max_length": arguments.max_length, "normalization_version": 1, "seed": 42,
                "trainable_encoder_layers": 2, "frozen_encoder_layers": 10, "epochs": arguments.epochs,
                "best_epoch": max(history, key=lambda entry: entry["validation_macro_f1"])["epoch"],
                "parameters": parameter_counts,
                "batch_size": arguments.batch_size, "optimizer": "AdamW", "learning_rate": 2e-5,
                "training_device": device, "inference_device": "cpu", "precision": "float32",
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "training_seconds": elapsed_before + time.perf_counter() - started}
    model.to("cpu").save_pretrained(BERT_DIRECTORY, safe_serialization=True)
    tokenizer.save_pretrained(BERT_DIRECTORY)
    report = {"schema_version": 1, "status": "trained", "metadata": metadata, "threshold": threshold,
              "validation_macro_f1": validation_f1, "history": history, "hardware": hardware_info(),
              "training_ids": selected_ids,
              "checkpoint_selection": "Best Macro-F1 on fixed stratified 2000-row validation subset. Final threshold on all validation rows. Test never used for training/selection."}
    write_json(BERT_REPORT, report)
    write_json(BERT_DIRECTORY / "training.json", {key: value for key, value in report.items() if key != "training_ids"})
    print(f"BERT training complete. Full-validation Macro-F1={validation_f1:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune multilingual BERT on a fixed training-only subset.")
    parser.add_argument("--train-rows", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1 or not 8 <= arguments.max_length <= 512:
        parser.error("epochs/batch-size must be positive; max-length must be in 8..512")
    train(arguments)
