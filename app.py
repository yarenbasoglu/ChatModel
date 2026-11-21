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


@st.cache_resource
def load_model(kind: str, filename: str, modified_ns: int):
    model_classes = {"baseline": BaselineModel, "neural": NeuralModel, "rules": EditDistanceModel,
                     "bert": BertModel, "bert_int8": QuantizedBertModel}
    return model_classes[kind].load(Path(filename))


st.title("ChatModel")
st.caption("SWE015 · Real-Time Chat Moderation")
st.info("İngilizce mesajlarla deneyin. Öğrenen modeller Jigsaw yorumlarıyla eğitildi; Türkçe doğruluğu değerlendirilmedi.")

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
    st.error("Eğitilmiş model bulunamadı. Veri indirildikten sonra hazırlama ve eğitim adımlarını çalıştırın.")
    st.code("python -m scripts.download_data\npython -m scripts.prepare_data\npython -m scripts.train_baseline")
    st.stop()

with st.sidebar:
    st.header("Model ve ölçümler")
    selected = st.selectbox("Çalışan model", list(available))

kind, filename = available[selected]
try:
    model = load_model(kind, str(filename), filename.stat().st_mtime_ns)
except Exception as error:
    st.error(f"Model yüklenemedi: {error}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    if kind == "rules":
        st.caption(f"{len(model.terms)} sözcüklük sözlük · Levenshtein uzaklığı ≤ 1")
        st.caption("Kısa sözcüklerde tam eşleşme aranır. Eşleşme skoru sözcük benzerliğidir; sınıf olasılığı hesaplanmaz.")
    else:
        st.caption(f"Eğitim örneği: {model.metadata['training_rows']:,}")
        st.caption(f"Karar eşiği: {model.threshold:.2f}")
    st.caption("CPU hedefi: 50 ms")
    if kind in ("bert", "bert_int8"):
        st.caption(f"En fazla {model.max_length} WordPiece token işlenir. Uzun mesajların sonu kesilebilir.")
    st.divider()
    if st.button("Sohbeti temizle", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.subheader("Mesaj gecikmeleri")
    if st.session_state.history:
        st.dataframe([
            {"Mesaj": index + 1, "Model": item["model_name"], "Sonuç": item["label"], "ms": round(item["latency_ms"], 2)}
            for index, item in enumerate(st.session_state.history)
        ], hide_index=True, use_container_width=True)
    else:
        st.caption("İlk mesajınızdan sonra burada görünür.")
    st.caption("Ölçüm: regex + seçili yöntem. Model yükleme ve ekran çizimi dahil değildir.")

if COMPARISON_REPORT.is_file():
    try:
        report = json.loads(COMPARISON_REPORT.read_text(encoding="utf-8"))
        with st.expander("Aynı test kümesindeki model karşılaştırması"):
            st.dataframe([
                {"Model": entry["model"], "Macro-F1": round(entry["test_metrics"]["macro_f1"], 4),
                 "Ortalama ms": round(entry["latency"]["mean_ms"], 3),
                 "p95 ms": round(entry["latency"]["p95_ms"], 3),
                 "Dosya MB": round(entry["artifact_size_mb"], 4)}
                for entry in report["models"]
            ], hide_index=True, use_container_width=True)
            st.caption(f"{report['dataset']['splits']['test']['rows']:,} ortak test örneği · Model başına 300 ölçüm · {report['hardware']['cpu']}")
            st.caption("Sözlük boyutu yalnızca kural yapılandırmasıdır. Sinir ağı boyutu Word2Vec vektörlerini de içerir. Kaydedilmiş karşılaştırma raporu gösterilir.")
            if any(entry["model"] == BertModel.name for entry in report["models"]):
                st.caption("BERT bu deneyde 12.000 eğitim mesajıyla, son iki katmanı ince ayarlanarak eğitildi. Tüm gecikmeler CPU'da ölçülür; toplu test skorunun cihazı raporda kayıtlıdır.")
    except (OSError, ValueError, KeyError):
        st.warning("Karşılaştırma raporu okunamadı. python -m scripts.evaluate_models çalıştırın.")
elif REPORT.is_file() and kind == "baseline":
    try:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        with st.expander("Gerçek test sonuçları ve veri kaynağı"):
            first, second, third = st.columns(3)
            first.metric("Test Macro-F1", f"{report['test_metrics']['macro_f1']:.4f}")
            second.metric("Ortalama CPU gecikmesi", f"{report['latency']['mean_ms']:.2f} ms")
            third.metric("Model dosyası", f"{report['artifact_size_mb']:.2f} MB")
            st.caption(f"{report['dataset']['splits']['test']['rows']:,} ayrılmış test örneği · {report['latency']['runs']} gecikme ölçümü · {report['hardware']['cpu']}")
            st.markdown("Veri: [Jigsaw / Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), [açık topluluk kopyası](https://huggingface.co/datasets/tasksource/jigsaw_toxicity).")
    except (OSError, ValueError, KeyError):
        st.warning("Test raporu okunamadı; rapor dosyasını kontrol edin.")

st.subheader("Sohbeti deneyin")
st.caption("Mesajı gönderdiğiniz anda analiz edilir. Offensive mesajlar tamamen maskelenir. Mesajlar ve gecikmeleri bu oturum boyunca tutulur.")
if kind != "rules":
    st.caption("Güven, seçilen sınıfın model olasılığıdır; ayrıca kalibre edilmedi ve doğruluk garantisi değildir.")

for item in st.session_state.history:
    with st.chat_message("user"):
        st.text(item["display_text"])
    with st.chat_message("assistant"):
        st.markdown(f"**{item['label']}** · {item['model_name']}")
        if item.get("score_kind") == "lexicon_similarity":
            st.caption(f"Sözlük eşleşmesi: %{item['confidence'] * 100:.1f} · Gecikme: {item['latency_ms']:.2f} ms")
        else:
            st.caption(f"Güven: %{item['confidence'] * 100:.1f} · Offensive olasılığı: %{item['offensive_probability'] * 100:.1f} · Gecikme: {item['latency_ms']:.2f} ms")

message = st.chat_input("İngilizce bir mesaj yazın…", max_chars=MAX_MESSAGE_LENGTH)
if message is not None:
    try:
        prediction = model.predict(message)
        item = asdict(prediction)
        item.pop("normalized_text")
        item["display_text"] = "[Saldırgan mesaj maskelendi]" if prediction.label == "Offensive" else message
        st.session_state.history.append(item)
        st.rerun()
    except ValueError as error:
        st.warning(str(error))
