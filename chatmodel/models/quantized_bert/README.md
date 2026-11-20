# Multilingual BERT INT8

İnce ayarlı BERT, ONNX'e aktarılır. Sabit matris çarpımı ağırlıkları ve embedding
tabloları dinamik INT8 quantization ile dönüştürülür. Orijinal tokenizer, 96 token
sınırı ve 0,61 karar eşiği kullanılır. Streamlit varsayılan olarak bu modeli seçer.
ONNX Runtime, Mac'te dört CPU iş parçacığı kullanır. Bu ayar 150 doğrulama
mesajında 1, 2 ve 4 iş parçacığı karşılaştırılarak seçildi. p95 sırasıyla
61,76, 35,20 ve 22,71 ms ölçüldü. Ayar test kümesindeki son ölçümden önce sabitlendi.

```bash
.venv/bin/python -m scripts.quantize_bert
.venv/bin/python -m scripts.evaluate_models --reuse-report reports/comparison.json
```

Model dosyası `artifacts/bert_int8/model.onnx` konumundadır. Dönüştürme komutu
mevcut model dizinini korur. Değerlendirme önceki satırları veri ve model
hash'leri eşleştiğinde kullanır. INT8 satırı aynı test kümesinde CPU'da ölçülür.
Ortak rapor `reports/comparison.json` dosyasındadır.

[ONNX Runtime quantization belgeleri](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
