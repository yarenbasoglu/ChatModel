import json

import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from transformers import BertConfig, BertForSequenceClassification

from chatmodel.models.bert import BertModel
from chatmodel.models.quantized_bert import QuantizedBertModel
from chatmodel.paths import QUANTIZED_BERT_MANIFEST
from chatmodel.paths import BERT_MANIFEST, PROCESSED, ROOT
from scripts.train_bert import freeze_encoder, select_subset


def test_training_subset_stays_inside_train(raw_frame):
    frame = raw_frame.rename(columns={"toxic": "label"})
    selected = select_subset(frame, 40)
    assert len(selected) == 40
    assert set(selected.id) <= set(frame.id)
    assert set(selected.label) == {0, 1}
    assert selected.id.tolist() == select_subset(frame, 40).id.tolist()


def test_encoder_freezing_updates_real_transformer_layers():
    configuration = BertConfig(vocab_size=30, hidden_size=16, num_hidden_layers=3,
                               num_attention_heads=2, intermediate_size=32, num_labels=2)
    model = BertForSequenceClassification(configuration)
    freeze_encoder(model, 2)
    assert not any(parameter.requires_grad for parameter in model.bert.embeddings.parameters())
    assert not any(parameter.requires_grad for parameter in model.bert.encoder.layer[0].parameters())
    assert all(parameter.requires_grad for parameter in model.bert.encoder.layer[-1].parameters())
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    original = model.bert.encoder.layer[-1].attention.self.query.weight.detach().clone()
    frozen = model.bert.embeddings.word_embeddings.weight.detach().clone()
    output = model(input_ids=torch.tensor([[2, 3, 4], [3, 4, 5]]), labels=torch.tensor([0, 1]))
    output.loss.backward()
    optimizer.step()
    assert not torch.equal(original, model.bert.encoder.layer[-1].attention.self.query.weight)
    assert torch.equal(frozen, model.bert.embeddings.word_embeddings.weight)


def test_incomplete_bert_is_rejected(tmp_path):
    (tmp_path / "training.json").write_text(json.dumps({"schema_version": 1, "status": "training"}))
    with pytest.raises(ValueError, match="tamamlanmamış"):
        BertModel.load(tmp_path)


@pytest.mark.skipif(not BERT_MANIFEST.is_file(), reason="Requires real fine-tuned BERT")
def test_real_bert_cpu_and_chat():
    model = BertModel.load(BERT_MANIFEST)
    manifest = json.loads((PROCESSED / "manifest.json").read_text())
    assert model.metadata["test_sha256"] == manifest["splits"]["test"]["sha256"]
    assert next(model.model.parameters()).device.type == "cpu"
    scores = model.batch_scores(["Thank you for helping", "You are a stupid idiot"])
    assert np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))
    with pytest.raises(ValueError):
        model.predict("!!!")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.selectbox[0].select(BertModel.name).run()
    app.chat_input[0].set_value("You are a stupid idiot and a fucking moron.").run()
    assert not app.exception
    last = app.session_state["history"][-1]
    assert last["model_name"] == BertModel.name
    assert last["label"] == "Offensive"
    assert last["display_text"] == "[Saldırgan mesaj maskelendi]"


@pytest.mark.skipif(not QUANTIZED_BERT_MANIFEST.is_file(), reason="Requires the real INT8 artifact")
def test_int8_is_default_and_masks_offensive_message():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.selectbox[0].value == QuantizedBertModel.name
    app.chat_input[0].set_value("Thank you for helping").run()
    app.chat_input[0].set_value("You are a stupid idiot and a fucking moron.").run()
    assert not app.exception
    clean, offensive = app.session_state["history"]
    assert clean["label"] == "Clean"
    assert offensive["model_name"] == QuantizedBertModel.name
    assert offensive["label"] == "Offensive"
    assert offensive["display_text"] == "[Saldırgan mesaj maskelendi]"
