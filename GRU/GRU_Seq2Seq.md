
# GRU Seq2Seq Model Sonuçları

## Değerlendirme Metrikleri (Test Seti Üzerinde)

Bu sonuçlar, modelin hiç görmediği 1000 adet test verisi üzerinde yapılan değerlendirme sonucunda elde edilmiştir.

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **BLEU** | 8.95 | Referans çeviri ile n-gram örtüşmesini ölçer. Düşük olması modelin tam kelime eşleşmesinde zorlandığını gösterir. |
| **METEOR** | 0.3289 | Eş anlamlıları ve kökleri de dikkate alır. BLEU'ya göre biraz daha esnektir. |
| **BERTScore (F1)** | 0.6156 | Anlamsal benzerliği ölçer. 0.61, çevirilerin bağlamsal olarak kısmen alakalı olduğunu ancak tam anlamı her zaman yakalayamadığını gösterir. |

---
**Model Dosyası:** `GRU/gru_seq2seq_model.pt`
