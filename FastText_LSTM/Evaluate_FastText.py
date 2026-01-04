
import torch
import torch.nn as nn
from datasets import load_from_disk
import numpy as np
import evaluate
import re
from collections import Counter
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# 1. Konfigürasyon
# ==========================================
DATASET_PATH = "open_subtitles_en_tr"
SRC_LANG = 'en'
TRG_LANG = 'tr'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
EMBED_DIM = 300
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
MAX_LEN = 20
MODEL_PATH = "fasttext_lstm_model.pt"

# ==========================================
# 2. Sınıflar (Evaluate için gerekli)
# ==========================================
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word2count = Counter()
        self.n_words = 4
    def add_sentence(self, sentence):
        for word in self.tokenize(sentence):
            self.word2count[word] += 1
    def tokenize(self, text):
        text = text.lower()
        parts = re.findall(r"[\w']+|[.,!?;]", text)
        return parts
    def build_vocab(self, max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ):
        common_words = self.word2count.most_common(max_size)
        for word, count in common_words:
            if count >= min_freq:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1
    def sentence_to_indices(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.word2index.get(token, self.word2index["<UNK>"]) for token in tokens]

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input, hidden, cell):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

# ==========================================
# 3. Değerlendirme
# ==========================================
def evaluate_model():
    print("Loading data...")
    raw_dataset = load_from_disk(DATASET_PATH)
    train_data = raw_dataset['train']
    
    # Vocab oluştur (Eğitimdeki ile aynı olmalı)
    # Hızlı olsun diye sample size aynen kullanılıyor
    SAMPLE_SIZE = 50000 
    if len(train_data) > SAMPLE_SIZE: 
        train_data_full = train_data.select(range(SAMPLE_SIZE))
    else:
        train_data_full = train_data

    print("Building Vocab...")
    src_vocab = Vocabulary(SRC_LANG)
    trg_vocab = Vocabulary(TRG_LANG)
    for item in train_data_full:
        src_vocab.add_sentence(item['translation'][SRC_LANG])
        trg_vocab.add_sentence(item['translation'][TRG_LANG])
    src_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    trg_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    
    # Değerlendirme için küçük bir set
    EVAL_SIZE = 1000
    if len(raw_dataset['train']) > 50000 + EVAL_SIZE:
        eval_data = raw_dataset['train'].select(range(50000, 50000 + EVAL_SIZE))
    else:
        eval_data = raw_dataset['train'].select(range(0, 100)) # Fallback
        
    print("Loading Model...")
    enc = Encoder(src_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(trg_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    metric_bleu = evaluate.load("sacrebleu")
    metric_meteor = evaluate.load("meteor")
    metric_bert = evaluate.load("bertscore")
    
    hypotheses = []
    references = []
    
    print("Generating translations...")
    with torch.no_grad():
        for i, item in enumerate(eval_data):
            src_text = item['translation'][SRC_LANG]
            trg_text = item['translation'][TRG_LANG]
            
            src_indices = [src_vocab.word2index["<SOS>"]] + src_vocab.sentence_to_indices(src_text) + [src_vocab.word2index["<EOS>"]]
            src_tensor = torch.tensor(src_indices, dtype=torch.long, device=DEVICE).unsqueeze(0)
            
            # Greedy Decode
            hidden, cell = model.encoder(src_tensor)
            input_tok = torch.tensor([trg_vocab.word2index["<SOS>"]], device=DEVICE)
            
            decoded_indices = []
            for _ in range(MAX_LEN):
                output, hidden, cell = model.decoder(input_tok, hidden, cell)
                top1 = output.argmax(1)
                input_tok = top1
                if top1.item() == trg_vocab.word2index["<EOS>"]:
                    break
                decoded_indices.append(top1.item())
            
            pred_words = [trg_vocab.index2word[idx] for idx in decoded_indices]
            hypotheses.append(" ".join(pred_words))
            references.append([trg_text])
            
            if i % 100 == 0: print(f"Processed {i}/{len(eval_data)}")

    print("Computing metrics (BLEU, METEOR & BERTScore)...")
    res_bleu = metric_bleu.compute(predictions=hypotheses, references=references)
    res_meteor = metric_meteor.compute(predictions=hypotheses, references=references)
    res_bert = metric_bert.compute(predictions=hypotheses, references=[r[0] for r in references], lang="tr")
    
    print(f"\n--- FastText LSTM Results ---")
    print(f"BLEU: {res_bleu['score']:.2f}")
    print(f"METEOR: {res_meteor['meteor']:.4f}")
    print(f"BERTScore F1: {np.mean(res_bert['f1']):.4f}")

if __name__ == "__main__":
    import nltk
    try: nltk.data.find("wordnet")
    except: nltk.download("wordnet")
    evaluate_model()
