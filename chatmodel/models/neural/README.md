# Word2Vec + Feedforward Neural Network

`python -m scripts.train_neural` sabit eğitim kümesinde kendi skip-gram
Word2Vec modelini eğitir. Boyut 100, pencere 5, min_count 3, negative sampling 5,
epoch 5 ve tek worker kullanılır. Hazır kelime vektörü indirilmez.

Mesajdaki bilinen kelime vektörlerinin ortalaması alınır. Bilinmeyen sözcükler
ortalama dışında bırakılır; tamamı bilinmeyen mesajın vektörü sıfırdır.
StandardScaler yalnızca train üzerinde öğrenilir.

Scikit-learn MLPClassifier gerçek feedforward ağı 100 → 128 ReLU → 32 ReLU →
1 sigmoid yapısındadır. SGD, momentum 0.9, öğrenme hızı 0.01, batch 256 ve
backpropagation kullanılır. Sınıf ağırlıkları yalnızca train dağılımından gelir.
En çok 30 epoch çalışır. Checkpoint ve karar eşiği aynı validation kümesindeki
Macro-F1 ile seçilir; 5 epoch iyileşme olmazsa eğitim durur. Ağırlıklar,
Word2Vec vektörleri, sözlük ve scaler `artifacts/neural.joblib` içinde saklanır.

Gensim 4.4.0'ın BLAS -1.0 hatası için `scripts.fix_gensim_blas` düzeltmesi
ve sayısal regresyon kontrolü bulunur. Ana README'deki yeniden eğitim adımlarını
izleyin. Derleme yalnızca ilk eğitim ortamını hazırlarken gerekir.

Eğitim günlüğü `reports/neural_training.json`, aynı test kümesindeki performans
ve 300 çağrılık CPU benchmark `reports/comparison.json` dosyasına yazılır.
