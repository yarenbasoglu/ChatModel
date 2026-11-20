import json

import pandas as pd
import pytest

from chatmodel.data import LABEL_COLUMNS
from chatmodel.models.baseline import BaselineModel, build_pipeline
from chatmodel.paths import COMPARISON_REPORT, PROCESSED


@pytest.fixture
def raw_frame():
    rows = []
    for index in range(100):
        offensive = index % 2
        row = {"id": f"id-{index}", "comment_text": f"{'stupid idiot awful' if offensive else 'thank you friendly helpful'} message {index}"}
        row.update({column: offensive if column == "toxic" else 0 for column in LABEL_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def fixture_model(raw_frame):
    pipeline = build_pipeline()
    pipeline.fit(raw_frame.comment_text, raw_frame.toxic)
    return BaselineModel(pipeline, 0.5, {"training_rows": len(raw_frame)})


@pytest.fixture
def dataset_manifest():
    local_manifest = PROCESSED / "manifest.json"
    if local_manifest.is_file():
        return json.loads(local_manifest.read_text(encoding="utf-8"))
    return json.loads(COMPARISON_REPORT.read_text(encoding="utf-8"))["dataset"]
