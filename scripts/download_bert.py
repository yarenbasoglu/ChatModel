from pathlib import Path
import os

from chatmodel.models.bert import BASE_MODEL, BASE_REVISION
from chatmodel.paths import ROOT


def download() -> Path:
    os.environ.setdefault("HF_HOME", str(ROOT / "data" / "raw" / "huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(ROOT / "data" / "raw" / "huggingface" / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    directory = ROOT / "data" / "raw" / "bert_base"
    snapshot_download(
        repo_id=BASE_MODEL, revision=BASE_REVISION, local_dir=directory,
        token=False,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
    )
    return directory


if __name__ == "__main__":
    print(download())
