import json

import pandas as pd
import pytest

from chatmodel.data import LABEL_COLUMNS, load_splits, make_splits, prepare_dataset, validate_raw


def test_binary_mapping_and_no_text_leakage(raw_frame):
    raw_frame.loc[0, "threat"] = 1
    repeated = raw_frame.iloc[[3]].copy()
    repeated["id"] = "duplicate"
    repeated["comment_text"] = repeated["comment_text"].str.upper()
    augmented = pd.concat([raw_frame, repeated], ignore_index=True)
    splits, summary = make_splits(augmented)
    assert summary["removed_duplicate_rows"] == 1
    all_rows = pd.concat(splits.values())
    assert all_rows.loc[all_rows.id == "id-0", "label"].item() == 1
    assert len(all_rows) == len(raw_frame)
    seen = set()
    for partition in splits.values():
        texts = set(partition.normalized_text)
        assert not texts & seen
        assert set(partition.label) == {0, 1}
        seen |= texts
    repeated_splits, _ = make_splits(augmented)
    assert splits["test"].id.tolist() == repeated_splits["test"].id.tolist()


def test_conflicting_duplicates_removed(raw_frame):
    conflict = raw_frame.iloc[[0]].copy()
    conflict["id"] = "conflict"
    conflict["insult"] = 1
    _, summary = make_splits(pd.concat([raw_frame, conflict], ignore_index=True))
    assert summary["removed_conflicting_rows"] == 2


@pytest.mark.parametrize("bad_label", [-1, 2, None])
def test_invalid_labels_rejected(raw_frame, bad_label):
    raw_frame["toxic"] = raw_frame["toxic"].astype(object)
    raw_frame.loc[0, "toxic"] = bad_label
    with pytest.raises(ValueError, match="binary"):
        validate_raw(raw_frame)


def test_incomplete_schema_rejected(raw_frame):
    with pytest.raises(ValueError, match="Missing"):
        validate_raw(raw_frame.drop(columns=LABEL_COLUMNS[0]))


def test_saved_splits_integrity_and_preservation(tmp_path, raw_frame):
    source = tmp_path / "source.csv"
    raw_frame.to_csv(source, index=False)
    destination = tmp_path / "processed"
    prepare_dataset(source, destination)
    splits, manifest = load_splits(destination)
    assert len(splits["train"]) == 80
    assert manifest["raw_rows"] == 100
    with pytest.raises(FileExistsError):
        prepare_dataset(source, destination)
    with (destination / "test.csv").open("a") as output:
        output.write("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_splits(destination)
