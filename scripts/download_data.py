import argparse
import json
import shutil
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import certifi

from chatmodel.data import sha256_file, validate_raw
from chatmodel.paths import RAW_DATA

REVISION = "d72c9a21492fe5db98cebb029f4fb0ec20e78e30"
DATA_URL = f"https://huggingface.co/datasets/tasksource/jigsaw_toxicity/resolve/{REVISION}/train.csv"


def download(destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Existing data preserved: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".csv.part")
    try:
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "ChatModel-course-project/1.0"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=120, context=context) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        frame = pd.read_csv(partial, dtype={"id": str})
        validate_raw(frame)
        if len(frame) != 159571:
            raise ValueError(f"Unexpected row count: {len(frame)}; expected the 159571-row Jigsaw training set.")
        digest = sha256_file(partial)
        partial.rename(destination)
    except (OSError, ValueError, urllib.error.URLError) as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Jigsaw download unavailable. Obtain train.csv from "
            "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data "
            f"and place it at {destination}. No substitute training data was generated. Cause: {error}"
        ) from error
    provenance = {
        "dataset": "Jigsaw Toxic Comment Classification Challenge (English)",
        "original_source": "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data",
        "download_source": DATA_URL, "mirror_revision": REVISION,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": digest, "rows": len(frame),
        "source_note": "Public community mirror; not an authenticated Kaggle download. Original Google dataset card describes annotations as CC0 and Wikipedia text as CC-BY-SA 3.0; mirror metadata lists Apache-2.0. Verify original terms before redistribution.",
    }
    destination.with_suffix(".source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description="Download real Jigsaw data from a pinned public mirror.")
    parser.add_argument("--output", type=Path, default=RAW_DATA)
    arguments = parser.parse_args()
    try:
        print(json.dumps(download(arguments.output), indent=2))
    except (RuntimeError, FileExistsError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
