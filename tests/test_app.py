from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from chatmodel.paths import ARTIFACT, ROOT


@pytest.mark.skipif(not ARTIFACT.is_file(), reason="Real-data integration test requires the trained baseline.")
def test_real_model_chat_masking_and_history():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "TF-IDF + Logistic Regression" in app.selectbox[0].options
    app.selectbox[0].select("TF-IDF + Logistic Regression").run()
    app.chat_input[0].set_value("Thank you for your help, have a wonderful day!").run()
    assert not app.exception
    assert len(app.session_state["history"]) == 1
    app.chat_input[0].set_value("You are a stupid idiot and a fucking moron.").run()
    assert not app.exception
    assert app.session_state["history"][-1]["label"] == "Offensive"
    assert app.session_state["history"][-1]["display_text"] == "[Saldırgan mesaj maskelendi]"
    assert all("fucking" not in element.value for element in app.text)
    assert len(app.dataframe) >= 1
    app.button[0].click().run()
    assert len(app.session_state["history"]) == 0


def test_missing_model_ui(monkeypatch, tmp_path):
    import chatmodel.paths

    monkeypatch.setattr(chatmodel.paths, "ARTIFACT", tmp_path / "missing.joblib")
    monkeypatch.setattr(chatmodel.paths, "NEURAL_ARTIFACT", tmp_path / "missing_neural.joblib")
    monkeypatch.setattr(chatmodel.paths, "RULE_CONFIG", tmp_path / "missing_rules.json")
    monkeypatch.setattr(chatmodel.paths, "BERT_MANIFEST", tmp_path / "missing_bert.json")
    monkeypatch.setattr(chatmodel.paths, "QUANTIZED_BERT_MANIFEST", tmp_path / "missing_int8.json")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "Eğitilmiş model bulunamadı" in app.error[0].value
    assert len(app.chat_input) == 0


def test_sidebar_retains_all_session_latencies():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.selectbox[0].select("Edit-distance sözlük kuralı").run()
    app.chat_input[0].set_value("Thanks for helping me!").run()
    first = app.session_state["history"][0]
    app.session_state["history"] = [first.copy() for _ in range(50)]

    app.chat_input[0].set_value("See you next round!").run()

    assert not app.exception
    assert len(app.session_state["history"]) == 51
    latency_table = app.sidebar.dataframe[0].value
    assert latency_table["Mesaj"].tolist() == list(range(1, 52))
    assert latency_table["ms"].iloc[0] == round(first["latency_ms"], 2)
    assert app.session_state["history"][0]["display_text"] == "Thanks for helping me!"
