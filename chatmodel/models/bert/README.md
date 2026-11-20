# Multilingual BERT

Temel checkpoint `google-bert/bert-base-multilingual-cased`, revision
`3f076fdb1ab68d5b2880cb87a0886f315b8146f8` kullanılır. İlk indirme yaklaşık 715 MB.

```bash
.venv/bin/python -m scripts.download_bert
.venv/bin/python -m scripts.train_bert --device mps
.venv/bin/python -m scripts.evaluate_models --bert-device mps
```

8 GB Apple M2 için ilk deney eğitim kümesinden seed 42 ile sınıf oranlarını
koruyan 12.000 örnek kullanır. İlk 10 encoder katmanı ve embeddingler dondurulur.
Son 2 katman, pooler ve sınıflandırma başlığı AdamW ile ince ayarlanır.
Öğrenme hızı 2e-5, batch 8, FP32, maksimum 96 WordPiece token, 2 epoch kullanılır.
Ortak regex normalizasyonu tokenizasyondan önce uygulanır. Bu deney İngilizce
etiketlerle eğitilir. Uzun mesajlar 96 token sınırında kesilir.

Checkpoint seçimi validation kümesinden sabit 2.000 örnekle yapılır. Son karar
eşiği validation kümesinin tamamında seçilir. Aynı 15.868 test örneği diğer
modellerle karşılaştırmada kullanılır. BERT'in eğitim örneği sayısı diğer
öğrenen modellerden azdır; karşılaştırma bu deney ayarlarını yansıtır.

Her 250 adımda yerel checkpoint kaydedilir. Kesilen eğitim aynı ayarlarla
`python -m scripts.train_bert --device mps --resume` komutuyla devam eder.
CPU eğitimi için `--device cpu` kullanılabilir. Hazır BERT dizinindeki
`training.json` tamamlanma kaydı oluşunca arayüzde BERT seçeneği görünür.

Uygulama ve 300 çağrılık gecikme ölçümü CPU'dadır. Toplu test sınıflandırması
`--bert-device mps` seçildiğinde GPU'da yapılır; ilk 32 test mesajının CPU
olasılıklarıyla fark ayrıca raporlanır. Tüm test sınıflandırmasını CPU'da
tekrarlamak için `--bert-device cpu` kullanın. Model boyutu tüm FP32 ağırlıklar,
tokenizer ve metadata dosyalarının toplamıdır. Quantization sonraki aşamadır.

Kaynak: https://huggingface.co/google-bert/bert-base-multilingual-cased
