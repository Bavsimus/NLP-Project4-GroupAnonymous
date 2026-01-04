
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
import random
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
import re

# ==========================================
# 1. Konfigürasyon ve Hiperparametreler
# ==========================================
DATASET_PATH = "open_subtitles_en_tr"
SRC_LANG = 'en'
TRG_LANG = 'tr'
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
EMBED_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 10
CLIP = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 20  # Çok uzun cümleleri filtrelemek için

# ==========================================
# 2. Veri Hazırlama ve Tokenization
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
        # Basit bir regex tokenizer
        text = text.lower()
        parts = re.findall(r"[\w']+|[.,!?;]", text)
        return parts

    def build_vocab(self, max_size=MAX_VOCAB_SIZE, min_freq=MIN_FREQ):
        # En çok geçen kelimeleri al
        common_words = self.word2count.most_common(max_size)
        
        for word, count in common_words:
            if count >= min_freq:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

    def sentence_to_indices(self, sentence):
        tokens = self.tokenize(sentence)
        return [self.word2index.get(token, self.word2index["<UNK>"]) for token in tokens]

class TranslationDataset(TorchDataset):
    def __init__(self, data, src_vocab, trg_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item['translation'][SRC_LANG]
        trg_text = item['translation'][TRG_LANG]
        
        src_indices = [self.src_vocab.word2index["<SOS>"]] + \
                      self.src_vocab.sentence_to_indices(src_text) + \
                      [self.src_vocab.word2index["<EOS>"]]
        
        trg_indices = [self.trg_vocab.word2index["<SOS>"]] + \
                      self.trg_vocab.sentence_to_indices(trg_text) + \
                      [self.trg_vocab.word2index["<EOS>"]]
                      
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    # Pad sequences
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True) # 0 is <PAD>
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# ==========================================
# 3. Model Tanımları (Encoder - Decoder - Seq2Seq)
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src)) 
        # embedded: [batch_size, src_len, emb_dim]
        
        outputs, hidden = self.gru(embedded)
        # outputs: [batch_size, src_len, hid_dim * n_directions]
        # hidden: [n_layers * n_directions, batch_size, hid_dim]
        
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        # input: [batch_size] (tek bir time step için token id'leri)
        # hidden: [n_layers, batch_size, hid_dim]
        
        input = input.unsqueeze(1) # [batch_size, 1]
        embedded = self.dropout(self.embedding(input)) 
        # embedded: [batch_size, 1, emb_dim]
        
        output, hidden = self.gru(embedded, hidden)
        # output: [batch_size, 1, hid_dim]
        
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Decoder çıktılarını saklamak için tensor
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder'dan son hidden state'i al
        hidden = self.encoder(src)
        
        # Decoder'a ilk giriş <SOS> tokenı (trg'nin ilk sütunu)
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t, :] = output
            
            # Teacher forcing: Bir sonraki input olarak gerçek hedefi mi yoksa tahmini mi kullanacağız?
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            
            input = trg[:, t] if teacher_force else top1
            
        return outputs

# ==========================================
# 4. Eğitim ve Yardımcı Fonksiyonlar
# ==========================================

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        # output: [batch_size, trg_len, output_dim]
        # trg: [batch_size, trg_len]
        
        output_dim = output.shape[-1]
        
        # Loss hesaplarken <SOS> tokenını atlıyoruz ve düzleştiriyoruz
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Batch {i} loss: {loss.item():.4f}")
        
    return epoch_loss / len(iterator)


import evaluate
import nltk
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")

def evaluate_model(model, iterator, src_vocab, trg_vocab, max_len=20):
    model.eval()
    metric_bleu = evaluate.load("sacrebleu")
    metric_meteor = evaluate.load("meteor")
    metric_bert = evaluate.load("bertscore")
    
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)
            
            # Greedy generation for GRU Seq2Seq
            batch_size = src.shape[0]
            # Encoder
            hidden = model.encoder(src)
            # Decoder
            input = torch.tensor([trg_vocab.word2index["<SOS>"]], device=DEVICE).repeat(batch_size)
            
            # Store predictions
            predictions = torch.zeros(batch_size, max_len).to(DEVICE)
            
            for t in range(max_len):
                # Decoder forward step
                output, hidden = model.decoder(input, hidden)
                top1 = output.argmax(1)
                predictions[:, t] = top1
                input = top1 # Next input is current prediction
            
            # Convert indices to words
            pred_indices = predictions.cpu().numpy()
            trg_indices = trg.cpu().numpy()
            
            for j in range(batch_size):
                pred_words = [trg_vocab.index2word[int(idx)] for idx in pred_indices[j] if int(idx) not in [0, 1, 2]]
                trg_words = [trg_vocab.index2word[int(idx)] for idx in trg_indices[j] if int(idx) not in [0, 1, 2]]
                
                hypotheses.append(" ".join(pred_words))
                references.append([" ".join(trg_words)])
            
            if len(hypotheses) > 100: break

    print("Computing metrics...")
    result_bleu = metric_bleu.compute(predictions=hypotheses, references=references)
    try:
        result_meteor = metric_meteor.compute(predictions=hypotheses, references=references)
    except:
        result_meteor = {"meteor": 0.0} # Fallback if nltk issue
        
    result_bert = metric_bert.compute(predictions=hypotheses, references=[r[0] for r in references], lang="tr")
    
    print(f"BLEU: {result_bleu['score']:.2f}")
    print(f"METEOR: {result_meteor['meteor']:.4f}")
    print(f"BERTScore F1: {np.mean(result_bert['f1']):.4f}")

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        raw_dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"Hata: Dataset yüklenemedi. Lütfen önce Dataset.py dosyasını çalıştırıp dataseti indirin. Hata detayı: {e}")
        return

    # Eğitim süresini kısaltmak için dataseti küçültebilirsiniz. Örn: dataset['train'].select(range(100000))
    train_data = raw_dataset['train']  # opus_open_subtitles genelde tek split döner ama splite dikkat et
    
    # Küçük bir örneklem alalım demo için (İsterseniz tamamını kullanın)
    SAMPLE_SIZE = 50000
    if len(train_data) > SAMPLE_SIZE:
        print(f"Dataset çok büyük, ilk {SAMPLE_SIZE} örnek alınıyor...")
        train_data = train_data.select(range(SAMPLE_SIZE))

    print("Building Vocabularies...")
    src_vocab = Vocabulary(SRC_LANG)
    trg_vocab = Vocabulary(TRG_LANG)

    for item in train_data:
        src_vocab.add_sentence(item['translation'][SRC_LANG])
        trg_vocab.add_sentence(item['translation'][TRG_LANG])
        
    src_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    trg_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    
    print(f"Source Vocab Size: {src_vocab.n_words}")
    print(f"Target Vocab Size: {trg_vocab.n_words}")
    
    print("Creating DataLoader...")
    dataset = TranslationDataset(train_data, src_vocab, trg_vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print("Initializing Model...")
    enc = Encoder(src_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    dec = Decoder(trg_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    model.apply(init_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is PAD
    
    print("Starting Training...")
    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | PPL: {np.exp(train_loss):.3f}')
        
    # Model kaydetme (opsiyonel)
    torch.save(model.state_dict(), 'gru_seq2seq_model.pt')
    print("Model saved!")
    
    print("Evaluating Model...")
    evaluate_model(model, dataloader, src_vocab, trg_vocab)


if __name__ == "__main__":
    main()
