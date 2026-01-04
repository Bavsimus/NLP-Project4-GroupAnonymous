
# Transformer Seq2Seq Model Sonuçları

## Değerlendirme Metrikleri (Test Seti Üzerinde)

Bu sonuçlar, modelin hiç görmediği 1000 adet test verisi üzerinde yapılan değerlendirme sonucunda elde edilmiştir.

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **BLEU** | 16.53 | GRU modeline göre (~8.95) neredeyse **iki kat** daha iyi performans göstermiştir. Bu, Transformer'ın kelime dizilimini ve yapısını çok daha iyi öğrendiğini kanıtlar. |
| **METEOR** | 0.4440 | Kelime kökü ve eş anlamlılık başarısı da GRU'ya göre belirgin şekilde yüksektir. |
| **BERTScore (F1)** | 0.6905 | Anlamsal yakınlık %69 seviyesindedir. Modelin ürettiği cümleler, hedef cümlenin anlamını büyük oranda korumaktadır. |

## Karşılaştırma Özeti

| Metric | GRU | Transformer | Fark |
|--------|-----|-------------|------|
| **BLEU** | 8.95 | **16.53** | +7.58 (Çok Büyük Fark) |
| **METEOR** | 0.3289 | **0.4440** | +0.1151 |
| **BERTScore** | 0.6156 | **0.6905** | +0.0749 |

**Sonuç:** Transformer modeli, tüm metriklerde GRU modeline üstünlük sağlamıştır.

---
**Model Dosyası:** `Transformers_Seq2Seq/transformer_seq2seq_model.pt`
