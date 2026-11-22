import json
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from chatmodel.models.baseline import BaselineModel
from chatmodel.models.neural import NeuralModel
from chatmodel.models.rules import EditDistanceModel
from chatmodel.models.bert import BertModel
from chatmodel.models.quantized_bert import QuantizedBertModel
from chatmodel.paths import QUANTIZED_BERT_MANIFEST
from chatmodel.paths import ARTIFACT, BERT_MANIFEST, COMPARISON_REPORT, MAX_MESSAGE_LENGTH, NEURAL_ARTIFACT, REPORT, RULE_CONFIG

st.set_page_config(page_title="ChatModel · Moderation", page_icon="💬", layout="wide")
st.html(Path(__file__).parent / ".streamlit" / "chatmodel.css")


@st.cache_resource
def load_model(kind: str, filename: str, modified_ns: int):
    model_classes = {"baseline": BaselineModel, "neural": NeuralModel, "rules": EditDistanceModel,
                     "bert": BertModel, "bert_int8": QuantizedBertModel}
    return model_classes[kind].load(Path(filename))


RULE_LABEL = "Edit-Distance Lexicon Rule"
MASKED_MESSAGE = "[Offensive message masked]"


def display_model_name(name: str) -> str:
    return RULE_LABEL if name == EditDistanceModel.name else name


with st.container(key="chatmodel_intro"):
    st.title("ChatModel", anchor=False)
    st.caption("SWE015 · Real-Time Chat Moderation")
st.info("Try an English message. The models were trained on English comments from Jigsaw.")

available = {}
for name, kind, filename in [
    (QuantizedBertModel.name, "bert_int8", QUANTIZED_BERT_MANIFEST),
    (BaselineModel.name, "baseline", ARTIFACT),
    (NeuralModel.name, "neural", NEURAL_ARTIFACT),
    (BertModel.name, "bert", BERT_MANIFEST),
    (EditDistanceModel.name, "rules", RULE_CONFIG),
]:
    if filename.is_file():
        available[name] = (kind, filename)

if not available:
    st.error("No trained model found. Download the released models or prepare the dataset and train a model.")
    st.code("python -m scripts.download_data\npython -m scripts.prepare_data\npython -m scripts.train_baseline")
    st.stop()

with st.sidebar:
    st.header("Models and Measurements")
    selected = st.selectbox("Active Model", list(available), format_func=display_model_name)

kind, filename = available[selected]
try:
    model = load_model(kind, str(filename), filename.stat().st_mtime_ns)
except Exception as error:
    st.error(f"Could not load the model. {error}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    if kind == "rules":
        st.caption(f"{len(model.terms)} words in the lexicon · Levenshtein distance ≤ 1")
        st.caption("Short words require an exact match. The score measures word similarity.")
    else:
        st.caption(f"Training examples: {model.metadata['training_rows']:,}")
        st.caption(f"Decision threshold: {model.threshold:.2f}")
    st.caption("CPU target: 50 ms")
    if kind in ("bert", "bert_int8"):
        st.caption(f"The model processes up to {model.max_length} WordPiece tokens. Longer messages may be truncated.")
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.subheader("Message Latencies")
    if st.session_state.history:
        st.dataframe([
            {"Message": index + 1, "Model": display_model_name(item["model_name"]), "Result": item["label"],
             "Latency (ms)": round(item["latency_ms"], 2)}
            for index, item in enumerate(st.session_state.history)
        ], hide_index=True, use_container_width=True)
    else:
        st.caption("Send a message to see its latency here.")
    st.caption("Timing covers text normalization and prediction. Model loading and screen rendering are excluded.")

if COMPARISON_REPORT.is_file():
    try:
        report = json.loads(COMPARISON_REPORT.read_text(encoding="utf-8"))
        with st.expander("Model Comparison on the Same Test Set"):
            st.dataframe([
                {"Model": display_model_name(entry["model"]),
                 "Macro-F1": round(entry["test_metrics"]["macro_f1"], 4),
                 "Mean latency (ms)": round(entry["latency"]["mean_ms"], 3),
                 "p95 latency (ms)": round(entry["latency"]["p95_ms"], 3),
                 "File size (MB)": round(entry["artifact_size_mb"], 4)}
                for entry in report["models"]
            ], hide_index=True, use_container_width=True)
            st.caption(
                f"{report['dataset']['splits']['test']['rows']:,} shared test messages. "
                f"300 single-message CPU predictions per model on {report['hardware']['cpu']}."
            )
            st.caption("The table shows saved evaluation results. Lexicon size covers the rule configuration file. Neural network size includes Word2Vec vectors.")
            if any(entry["model"] == BertModel.name for entry in report["models"]):
                st.caption("BERT's final two encoder layers were fine-tuned on 12,000 training messages. All latency measurements used the CPU. The report lists the device used to compute each test score.")
    except (OSError, ValueError, KeyError):
        st.warning("Could not read the comparison report. Run python -m scripts.evaluate_models.")
elif REPORT.is_file() and kind == "baseline":
    try:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        with st.expander("Test Results and Data Source"):
            first, second, third = st.columns(3)
            first.metric("Test Macro-F1", f"{report['test_metrics']['macro_f1']:.4f}")
            second.metric("Mean CPU Latency", f"{report['latency']['mean_ms']:.2f} ms")
            third.metric("Model File Size", f"{report['artifact_size_mb']:.2f} MB")
            st.caption(f"{report['dataset']['splits']['test']['rows']:,} held-out test messages · {report['latency']['runs']} latency measurements · {report['hardware']['cpu']}")
            st.markdown("Data: [Jigsaw / Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), [public dataset mirror](https://huggingface.co/datasets/tasksource/jigsaw_toxicity).")
    except (OSError, ValueError, KeyError):
        st.warning("Could not read the test report.")

st.subheader("Try the Chat")
st.caption("Messages are analyzed as you send them. Offensive messages are masked. Chat history and latency records are kept for this session.")
if kind != "rules":
    st.caption("Confidence is the model's estimated probability for the predicted label.")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.text(MASKED_MESSAGE if item["label"] == "Offensive" else item["display_text"])
    with st.chat_message("assistant"):
        badge_color = "red" if item["label"] == "Offensive" else "green"
        st.markdown(f":{badge_color}-badge[{item['label']}] · {display_model_name(item['model_name'])}")
        if item.get("score_kind") == "lexicon_similarity":
            st.caption(f"Lexicon match: {item['confidence'] * 100:.1f}% · Latency: {item['latency_ms']:.2f} ms")
        else:
            st.caption(f"Confidence: {item['confidence'] * 100:.1f}% · Offensive probability: {item['offensive_probability'] * 100:.1f}% · Latency: {item['latency_ms']:.2f} ms")

message = st.chat_input("Type a message in English…", max_chars=MAX_MESSAGE_LENGTH)
if message is not None:
    try:
        prediction = model.predict(message)
        item = asdict(prediction)
        item.pop("normalized_text")
        item["display_text"] = MASKED_MESSAGE if prediction.label == "Offensive" else message
        st.session_state.history.append(item)
        st.rerun()
    except ValueError as error:
        st.warning(str(error))
