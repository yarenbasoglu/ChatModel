import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from chatmodel.models.neural import NeuralModel, average_embeddings
from chatmodel.models.rules import EditDistanceModel
from chatmodel.paths import NEURAL_ARTIFACT, ROOT, RULE_CONFIG
from scripts.fix_gensim_blas import verify_backend


@pytest.fixture
def rules():
    return EditDistanceModel.load(RULE_CONFIG)


@pytest.mark.parametrize("message", ["you idiot", "you idiiot", "stup1d", "you are stuupid"])
def test_exact_noisy_and_edit_matches(rules, message):
    prediction = rules.predict(message)
    assert prediction.label == "Offensive"
    assert prediction.offensive_probability is None
    assert prediction.score_kind == "lexicon_similarity"
    assert prediction.confidence > 0


@pytest.mark.parametrize("message", ["classical music", "a shirt", "Thank you for helping", "assumption"])
def test_whole_words_and_short_token_guard(rules, message):
    assert rules.predict(message).label == "Clean"


@pytest.mark.parametrize("message", ["", "!!!", "a" * 5001])
def test_rules_input_limits(rules, message):
    with pytest.raises(ValueError):
        rules.predict(message)


def test_embedding_average_preserves_repetitions_and_handles_oov():
    vocabulary = {"hello": 0, "world": 1}
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
    result = average_embeddings(["HELLO hello world unknown", "unseen"], vocabulary, vectors)
    np.testing.assert_allclose(result[0], [2 / 3, 1 / 3])
    np.testing.assert_array_equal(result[1], [0, 0])
    assert np.isfinite(result).all()


def test_word2vec_negative_dot_product_regression():
    verify_backend()


def test_rule_switch_and_mixed_history():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.selectbox[0].select("TF-IDF + Logistic Regression").run()
    app.chat_input[0].set_value("Thank you for helping!").run()
    app.selectbox[0].select(EditDistanceModel.name).run()
    app.chat_input[0].set_value("you idiiot").run()
    assert not app.exception
    history = app.session_state["history"]
    assert history[0]["model_name"] == "TF-IDF + Logistic Regression"
    assert history[-1]["model_name"] == EditDistanceModel.name
    assert history[-1]["display_text"] == "[Offensive message masked]"
    assert history[-1]["offensive_probability"] is None
    assert any("Lexicon match" in item.value for item in app.caption)
    app.chat_input[0].set_value("!!!").run()
    assert len(app.session_state["history"]) == 2
    assert app.warning


@pytest.mark.skipif(not NEURAL_ARTIFACT.is_file(), reason="Requires genuinely trained neural weights")
def test_real_neural_and_ui(dataset_manifest):
    model = NeuralModel.load(NEURAL_ARTIFACT)
    prediction = model.predict("Thank you for helping!")
    assert 0 <= prediction.offensive_probability <= 1
    assert prediction.latency_ms > 0
    assert model.metadata["training_rows"] == dataset_manifest["splits"]["train"]["rows"]
    assert model.metadata["test_sha256"] == dataset_manifest["splits"]["test"]["sha256"]
    with pytest.raises(ValueError):
        model.predict("!!!")
    assert model.classifier.hidden_layer_sizes == (128, 32)
    assert model.classifier.solver == "sgd"
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.selectbox[0].select(NeuralModel.name).run()
    app.chat_input[0].set_value("You are a stupid idiot and a fucking moron.").run()
    assert not app.exception
    history = app.session_state["history"]
    assert history[-1]["model_name"] == NeuralModel.name
    assert history[-1]["label"] == "Offensive"
    assert history[-1]["display_text"] == "[Offensive message masked]"
