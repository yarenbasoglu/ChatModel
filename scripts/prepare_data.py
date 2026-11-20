import argparse
import json
from pathlib import Path

from chatmodel.data import prepare_dataset
from chatmodel.paths import PROCESSED, RAW_DATA


def main():
    parser = argparse.ArgumentParser(description="Create fixed, deduplicated 80/10/10 Jigsaw splits.")
    parser.add_argument("--input", type=Path, default=RAW_DATA)
    parser.add_argument("--output", type=Path, default=PROCESSED)
    arguments = parser.parse_args()
    print(json.dumps(prepare_dataset(arguments.input, arguments.output), indent=2))


if __name__ == "__main__":
    main()
