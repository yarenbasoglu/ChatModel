# İlk sürüm doğrulaması

## BERT güncellemesi — 15 Kasım 2026

Gerçek mBERT checkpoint'i 12.000 Jigsaw train örneğiyle, son iki encoder
katmanı ve sınıflandırma başlığı güncellenerek 2 epoch eğitildi. En iyi epoch 2.
Tam validation Macro-F1 0,856924, eşik 0,61. Ortak test Macro-F1 0,844827.
Tüm test sınıflandırması MPS'de; 300 tek-mesaj gecikmesi CPU'da ölçüldü.
Ortalama 45,012 ms, p95 62,253 ms. p95 50 ms hedefini aşıyor.
32 örneğin CPU/MPS karşılaştırmasında tüm kararlar aynı, en büyük olasılık
farkı 8,79e-7. Nihai dosyalar `artifacts/bert/` içinde, boyut 715,363 MB.

49 test geçti. Gerçek BERT ile Streamlit AppTest üzerinden seçim, gönderim
ve Offensive maskeleme doğrulandı. Sunucu sağlık kontrolü `ok` döndürdü.
Görsel tarayıcı incelemesi yapılmadı. Öğrenme ve test ayrımı sabit hash'lerle
korundu. BERT eğitim alt kümesi kullanıldığı README ve arayüzde belirtilir.

## İkinci aşama güncellemesi

14 Kasım 2026 tarihinde Word2Vec + Neural Network ve edit-distance sözlük
kuralı eklendi. Word2Vec gerçek 126.941 train mesajında eğitildi. Sinir ağı
100 → 128 → 32 → 1 yapısında, SGD/backpropagation ile 18 epoch çalıştı;
validation'a göre 13. epoch seçildi. Ortak test raporu `comparison.json` içindedir.

- `pytest -q -p no:cacheprovider`: 45 passed in 1.53s, skip yok.
- `pip check`: No broken requirements found.
- HTTP `/_stcore/health`: `ok`.
- AppTest üzerinden üç yöntem arasında seçim ve gerçek sinir ağıyla maskeleme başarılı.
- Gensim 4.4.0 BLAS hatası proje içindeki yeniden derleme script'iyle düzeltildi;
  -1.0 iç çarpım regresyonu test edildi. Hatalı başlangıç eğitimi durdurulup
  düzeltme sonrası gerçek eğitim baştan çalıştırıldı.
- Eğitim sonunda CPU modelini sorgulayan çağrı izin hatası verdi. Ağırlıklar
  başarıyla kaydedilmişti. Eğitim raporu kaydedilen ağırlık metadata'sı ve
  konsolda gözlenen 18 epoch kaydıyla tamamlandı. CPU sorgusunun raporlamayı
  durdurmaması için hata yakalama eklendi. Ortak ölçümde Apple M2 bilgisi doğrulandı.
- TF-IDF ağırlığının SHA-256 değeri ilk sürümle aynı.
- BERT ve quantized BERT sonraki aşamalar olarak duruyor.

## İlk çalıştırmanın kaydı

Tarih: 14 Kasım 2026 (Europe/Istanbul).

- Python 3.13.3 ortamı: `.venv`.
- `python -m pip check`: No broken requirements found.
- `python -m pytest -q -p no:cacheprovider`: 30 passed in 0.99s; skip yok.
- Streamlit sunucusu: `http://127.0.0.1:8501`.
- `/_stcore/health`: HTTP 200, `ok`.
- Model: gerçek Jigsaw verisinden TF-IDF + Logistic Regression.
- Eğitim, split ve performans kanıtları: `baseline_metrics.json`.
- AppTest: gerçek model yükleme, canlı girdi, maskeleme, yan panel geçmişi ve
  sohbet temizleme başarılı; model yokken açık hata gösterimi de doğrulandı.
- Görsel tarayıcı bağlantısı zaman aşımına uğradı; screenshot/görsel QA yapılmadı.
- Neural Network, BERT ve quantized BERT eğitilmedi; yalnızca sonraki aşama
  dizinleri, ortak tahmin arayüzü ve uygulama planları hazırlandı.

Elle verilen İngilizce smoke girdilerinde `Thank you for your help, have a
wonderful day!` Clean (%99,52 sınıf skoru), açık hakaret içeren test cümlesi
Offensive (yaklaşık %100 sınıf skoru), `You are stuuuupid` Offensive (%99,64)
olarak döndü. Bunlar eğitim ya da test skorunu değiştirmek için kullanılmadı.
Tekil smoke çağrısının ilk gecikmesi 7,59 ms ölçüldü; rapordaki 0,461 ms sıcak
ortalama ile karıştırılmamalıdır. Model olasılıkları ayrıca kalibre edilmedi.
