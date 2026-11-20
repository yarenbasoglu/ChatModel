import joblib
import pytest
import sklearn

from chatmodel.evaluation import benchmark, choose_threshold, classification_metrics
from chatmodel.models.baseline import BaselineModel


def test_predict_probability_and_latency(fixture_model):
    clean = fixture_model.predict("thank you friendly helpful")
    offensive = fixture_model.predict("stupid idiot awful")
    assert clean.label == "Clean"
    assert offensive.label == "Offensive"
    assert 0 <= clean.confidence <= 1
    assert clean.confidence == 1 - clean.offensive_probability
    assert offensive.confidence == offensive.offensive_probability
    assert offensive.latency_ms > 0


@pytest.mark.parametrize("text", ["", "  ", "!!!", "123456", "https://example.com", "a" * 5001, None])
def test_invalid_messages(fixture_model, text):
    with pytest.raises(ValueError):
        fixture_model.predict(text)


def test_round_trip_preserves_prediction(tmp_path, fixture_model):
    filename = tmp_path / "fixture_only.joblib"
    joblib.dump({"schema_version": 1, "status": "trained", "pipeline": fixture_model.pipeline,
                 "threshold": 0.5, "metadata": {"sklearn_version": sklearn.__version__}}, filename)
    loaded = BaselineModel.load(filename)
    assert loaded.predict("stupid idiot").offensive_probability == fixture_model.predict("stupid idiot").offensive_probability


def test_missing_model_is_not_faked(tmp_path):
    with pytest.raises(FileNotFoundError):
        BaselineModel.load(tmp_path / "missing.joblib")


def test_validation_threshold_and_metrics():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.4, 0.9]
    threshold, score = choose_threshold(labels, probabilities)
    assert 0.2 < threshold <= 0.4
    assert score == 1
    assert classification_metrics(labels, probabilities, threshold)["confusion_matrix"] == [[2, 0], [0, 2]]


def test_latency_uses_at_least_100_predictions(fixture_model):
    result = benchmark(fixture_model, ["thank you"], runs=100)
    assert result["runs"] == 100
    assert 0 < result["mean_ms"] <= result["max_ms"]
    with pytest.raises(ValueError):
        benchmark(fixture_model, ["thank you"], runs=99)
