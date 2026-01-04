# FastText + LSTM Seq2Seq Model Sonuçları

## Model Mimarisi
Bu model, kelime temsilleri için **Pre-trained FastText Embeddings** (Wiki-News-300) ve diziden diziye öğrenme için **LSTM** mimarisini kullanmıştır.
*   **Embedding:** FastText (300 boyutlu, statik)
*   **Encoder/Decoder:** 2 Katmanlı LSTM
*   **Kelime Dağarcığı:** 20.000 (En Sık Kullanılanlar)

## Değerlendirme Metrikleri (Test Seti Üzerinde)

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **BLEU** | 3.03 | CNN (~2.74) ile benzer, GRU ve Transformer'dan oldukça düşüktür. Hazır embedding kullanılmasına rağmen modelin cümle yapısını kurmakta zorlandığı görülmektedir. |
| **METEOR** | 0.2404 | Düşük bir skor. Kelime eşleşmelerinin başarısız olduğunu gösterir. |
| **BERTScore (F1)** | 0.5029 | CNN'den (~0.40) daha iyi, ancak GRU (~0.61) ve Transformer'dan (~0.69) düşüktür. FastText embedding'leri sayesinde kelimelerin anlamsal yakınlığı bir miktar yakalanmıştır ancak cümle bağlamı zayıftır. |

## Karşılaştırma ve Analiz
*   **FastText Etkisi:** Pre-trained embedding kullanımı, BERTScore'un CNN modeline kıyasla yükselmesini sağlamıştır (%10 artış). Bu, kelime bazlı anlamsal özelliklerin daha iyi temsil edildiğini gösterir.
*   **LSTM vs GRU:** Daha önce eğittiğimiz GRU modelinin (BLEU ~9) çok daha başarılı olması, **"Sıfırdan eğitilen (learnable) embedding"lerin**, genel amaçlı pre-trained embedding'lere (FastText) göre bu veri seti özelinde daha iyi uyum sağladığını düşündürmektedir.
*   **Sonuç:** Sadece embedding değiştirmek performansı artırmaya yetmemiştir; modelin dikkat mekanizması (attention) eksikliği ve LSTM'in uzun cümlelerdeki unutma sorunu devam etmektedir.

**Model Dosyası:** `fasttext_lstm_model.pt`
