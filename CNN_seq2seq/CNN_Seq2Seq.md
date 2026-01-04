# CNN Seq2Seq Model Sonuçları

## Model Mimarisi: Fully Convolutional Seq2Seq (ConvS2S)
Bu model, klasik RNN veya Transformer yerine **Convolutional Neural Networks (CNN)** kullanılarak eğitilmiştir. Özellikle yerel (local) özelliklerin çıkarılmasında etkili olan bu mimari, dil çevirisi görevinde test edilmiştir.

## Değerlendirme Metrikleri (Test Seti Üzerinde)

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **BLEU** | 2.74 | Diğer modellere kıyasla oldukça düşüktür. CNN mimarisi, uzun cümlelerdeki ilişkileri (long-range dependencies) yakalamakta RNN ve Transformer kadar başarılı olamamış olabilir. |
| **METEOR** | 0.1963 | Kelime kökü ve anlam eşleşmesi başarısı sınırlı kalmıştır. |
| **BERTScore (F1)** | 0.4041 | Anlamsal benzerlik skoru da düşüktür; üretilen çeviriler hedef cümlenin anlamından uzaktır. |

## Analiz
*   CNN tabanlı Seq2Seq modeli, bu veri seti ve hiperparametrelerle **Transformer ve GRU'nun gerisinde kalmıştır.**
*   Bunun olası sebepleri:
    *   CNN'in 'Receptive Field' (Görüş Alanı) sınırlıdır; çok katmanlı yapıya rağmen cümlenin başı ile sonu arasındaki ilişkiyi kurmakta zorlanmış olabilir.
    *   Transformer'daki Self-Attention veya RNN'deki Hidden State mekanizması, dil modellemede Convolution'a göre daha doğal bir avantaj sağlar.
    *   Hiperparametre optimizasyonu (Kernel Size, Katman Sayısı) bu model için daha kritik ve zordur.

**Model Dosyası:** `cnn_seq2seq_model.pt`
