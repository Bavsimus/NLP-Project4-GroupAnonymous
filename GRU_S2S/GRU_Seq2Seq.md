# GRU Seq2Seq Model Eğitim Sonuçları

Bu dosya, İngilizce-Türkçe çeviri görevi için eğitilen GRU tabanlı Seq2Seq modelinin eğitim sonuçlarını ve performans metriklerini içermektedir.

## 1. Model ve Veri Özellikleri

- **Model Mimarisi:** GRU (Gated Recurrent Unit) tabanlı Seq2Seq (Encoder-Decoder)
- **Veri Seti:** OpenSubtitles (İngilizce - Türkçe)
- **Kullanılan Örnek Sayısı:** 50,000 (Eğitim süresini optimize etmek için alt küme kullanıldı)
- **Kaynak Dil (İngilizce) Kelime Dağarcığı:** 7,536 kelime
- **Hedef Dil (Türkçe) Kelime Dağarcığı:** 16,499 kelime
- **Hiperparametreler:**
  - Embedding Boyutu: 256
  - Hidden Boyutu: 512
  - Katman Sayısı (Layers): 2
  - Dropout: 0.5
  - Batch Size: 64
  - Learning Rate: 0.001
  - Optimizer: Adam
  - Epoch Sayısı: 10

## 2. Eğitim Süreci ve Kayıp (Loss) Değerleri

Model 10 epoch boyunca eğitilmiş ve her epoch sonunda Eğitim Kaybı (Train Loss) ve Perplexity (PPL) değerleri kaydedilmiştir.

| Epoch | Train Loss | Perplexity (PPL) | Açıklama |
|-------|------------|------------------|----------|
| 01 | 5.460 | 235.052 | Başlangıç aşaması, model öğrenmeye yeni başlıyor. |
| 02 | 4.320 | 75.186 | PPL değerinde ciddi bir düşüş, hızlı öğrenme. |
| 03 | 3.534 | 34.277 | Model dil yapısını kavramaya başlıyor. |
| 04 | 2.957 | 19.239 | Kararlı düşüş devam ediyor. |
| 05 | 2.527 | 12.518 | Hata oranı azalıyor. |
| 06 | 2.201 | 9.037 | PPL tek haneli değerlere yaklaşıyor. |
| 07 | 1.946 | 7.002 | İnce ayar öğrenme aşaması. |
| 08 | 1.751 | 5.762 | Model çıktılarında güven artıyor. |
| 09 | 1.599 | 4.948 | Çok düşük belirsizlik seviyesi. |
| 10 | **1.481** | **4.399** | Final performansı. Oldukça başarılı bir yakınsama. |

## 3. Eğitim Grafiği Özeti

- **Loss Düşüşü:** Eğitim kaybı 9.75 seviyelerinden başlayıp (ilk batch), düzenli bir şekilde düşerek 1.48 seviyesine inmiştir. Bu, modelin veriye başarılı bir şekilde overfit olmadan (veya kontrollü bir şekilde) uyum sağladığını gösterir.
- **Perplexity (PPL):** Başlangıçtaki 235 seviyesinden 4.4 seviyesine inmesi, modelin bir sonraki kelimeyi tahmin etme başarısının çarpıcı bir şekilde arttığını gösterir. Düşük PPL, modelin çeviri konusunda kendine daha fazla güvendiğini işaret eder.

## 4. Sonuç ve Değerlendirme

- Model, 50.000 veriyle sınırlandırılmış bir eğitim setinde bile 10 epoch sonunda **4.399** gibi oldukça düşük bir Perplexity değerine ulaşmıştır.
- Türkçe'nin sondan eklemeli yapısı nedeniyle hedef kelime dağarcığının (16,499) kaynak dile göre (7,536) çok daha geniş olmasına rağmen modelin başarılı olduğu görülmektedir.
- Bu sonuçlar, modelin temel çeviri kalıplarını öğrendiğini ve verilen cümlelere anlamlı (en azından eğitim seti dahilinde) karşılıklar üretebilecek kapasiteye geldiğini göstermektedir.
- Daha iyi genelleme (generalization) için veri seti boyutu artırılabilir veya daha uzun süreli eğitim yapılabilir, ancak mevcut haliyle kavram kanıtı (PoC) olarak başarılıdır.

**Model Dosyası:** `gru_seq2seq_model.pt` olarak kaydedilmiştir.
