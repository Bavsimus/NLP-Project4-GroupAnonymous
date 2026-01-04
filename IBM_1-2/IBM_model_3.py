import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import string
from collections import defaultdict


try:
    # Arkadaşının hazırladığı IBMModel2 sınıfını miras alıyoruz
    from IBM_model_1and2 import IBMModel2 
except ImportError:
    print("Hata: 'IBM_model_1and2.py' bulunamadı. Lütfen dosyanın IBM_1-2 klasöründe olduğundan emin olun.")

class IBMModel3(IBMModel2):
    def __init__(self, data):
        """
        IBM Model 3: Fertility (Doğurganlık) ve Distortion parametrelerini ekler.
        """
        super().__init__(data)
        # n(phi | e): Bir İngilizce kelimenin kaç Türkçe kelime üreteceği
        self.n_probs = defaultdict(lambda: 1.0 / 3) 
        # d(j | i, l, m): Kelime dizilim olasılığı
        self.d_probs = defaultdict(lambda: 1.0)

    def train_model3(self, iterations=3):
        print(f">>> IBM Model 3 Eğitiliyor ({iterations} iterasyon)...")
        for it in range(iterations):
            count_tr_en = defaultdict(float)
            total_en = defaultdict(float)
            count_fertility = defaultdict(float)
            total_fertility = defaultdict(float)

            for en_tokens, tr_tokens in self.corpus:
                l, m = len(en_tokens) - 1, len(tr_tokens)
                
                # Viterbi Hizalama (Model 2'den gelen t_probs ve a_probs kullanılır)
                alignment = []
                for j, tr_word in enumerate(tr_tokens):
                    best_prob, best_i = -1, 0
                    for i, en_word in enumerate(en_tokens):
                        prob = self.t_probs[(tr_word, en_word)] * self.a_probs.get((i, j, l, m), 1.0/(l+1))
                        if prob > best_prob:
                            best_prob, best_i = prob, i
                    alignment.append(best_i)

                phi_counts = defaultdict(int)
                for j, i in enumerate(alignment):
                    count_tr_en[(tr_tokens[j], en_tokens[i])] += 1
                    total_en[en_tokens[i]] += 1
                    phi_counts[i] += 1

                for i, en_word in enumerate(en_tokens):
                    count_fertility[(phi_counts[i], en_word)] += 1
                    total_fertility[en_word] += 1

            # M-Step: Güncelleme
            for (tr, en), val in count_tr_en.items():
                self.t_probs[(tr, en)] = val / total_en[en]
            for (f, en), val in count_fertility.items():
                self.n_probs[(f, en)] = val / total_fertility[en]
            print(f"İterasyon {it+1} tamamlandı.")

if __name__ == "__main__":
    try:
        import importlib
        acq_pre = importlib.import_module("acquisition-preprocessing")
        train_data = acq_pre.train_data
        
        # Modeli başlat ve eğit
        model = IBMModel3(train_data[:2000]) # Test için küçük veri
        model.train(iterations=2)            # Önce Model 2 eğitimi
        model.train_model3(iterations=2)     # Senin Model 3 eğitimin
        
        print("\n" + "="*40)
        print("BAŞARILI: IBM Model 3 hazırlandı!")
        print("="*40)
        
    except ImportError as e:
        print(f"Hata: Gerekli dosyalar yüklenemedi. Detay: {e}")