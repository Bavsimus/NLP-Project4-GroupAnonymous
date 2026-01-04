# IBM Model 3: İstatistiksel Makine Çevirisi ve Doğurganlık Modellemesi

Bu rapor, projenin **IBM Model 3** aşamasında gerçekleştirilen teknik geliştirmeleri, kullanılan metrikleri ve modelin matematiksel altyapısını detaylandırmaktadır.

## 1. Proje Özeti
IBM Model 1 ve 2, kelimeler arasında birebir (1-to-1) eşleşme olduğunu varsayar. Ancak İngilizce-Türkçe gibi diller arasında bir kelime bazen birden fazla kelimeye karşılık gelebilir (Örn: "home" -> "ev", "study" -> "ders çalışmak"). Model 3, bu sorunu **Fertility (Doğurganlık)** kavramıyla çözer.

## 2. Teknik Özellikler ve Parametreler

Model 3 kapsamında aşağıdaki üç temel olasılık dağılımı eğitilmiştir:

| Parametre | Teknik Adı | Açıklama |
| :--- | :--- | :--- |
| **t(f\|e)** | Translation Probability | Sözlüksel çeviri olasılığı (Model 1 & 2'den miras). |
| **n(φ\|e)** | Fertility | Bir kaynak kelimenin (e), hedef dilde kaç kelime (φ) üreteceği. |
| **d(j\|i, l, m)** | Distortion | Kelimenin cümle içindeki konumunun ne kadar kayacağı (Yer değiştirme). |

### Kullanılan Algoritmalar:
- **Expectation-Maximization (EM):** Eksik verileri (hizalamaları) tahmin etmek ve parametreleri optimize etmek için kullanıldı.
- **Viterbi Alignment:** Model 3'ün karmaşık yapısı nedeniyle, en olası hizalama dizisini bulmak için Viterbi yaklaşımı tercih edilmiştir.

## 3. Veri Seti ve Ön İşleme
- **Kaynak:** Tatoeba English-Turkish Dataset.
- **Eğitim Seti:** 27,000 Cümle.
- **Test Seti:** 3,000 Cümle.
- **Preprocessing:** Noktalama işaretleri temizlendi (String manipulation), tüm metin küçük harfe (Lowercasing) çevrildi ve NULL token eklemesi yapıldı.

## 4. Değerlendirme Metrikleri

Modelin başarısı aşağıdaki metriklerle ölçülmektedir:

1. **Alignment Error Rate (AER):** Üretilen hizalamaların referans hizalamalarla ne kadar örtüştüğünü ölçer.
2. **Perplexity:** Modelin veriyi ne kadar iyi tahmin ettiğini gösterir (Daha düşük değer daha iyidir).
3. **Fertility Score:** Bir kelimenin ürettiği kelime sayısı dağılımının doğruluğu.

## 5. Model Çıktı Analizi (Örnek)

Eğitim sonrası elde edilen örnek bir doğurganlık (fertility) dağılımı:
- `house` -> 1 kelime üretme olasılığı: `0.88`
- `house` -> 0 kelime (silinme) olasılığı: `0.02`
- `not` -> 2 kelime üretme olasılığı: `0.15` (Örn: "-me", "-ma" ekleri ile birleştiğinde)

## 6. Kurulum ve Çalıştırma
Gereksinimleri yüklemek için:
```bash
pip install -r requirements.txt