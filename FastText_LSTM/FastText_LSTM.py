
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter
import evaluate
import gensim.downloader as api
import os

# ==========================================
# 1. Konfigürasyon
# ==========================================
DATASET_PATH = "open_subtitles_en_tr"
SRC_LANG = 'en'
TRG_LANG = 'tr'
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
EMBED_DIM = 300  # FastText genelde 300 boyutludur
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
N_EPOCHS = 10
CLIP = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 20

# ==========================================
# 2. Veri Hazırlama & Pre-trained Embedding
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

def load_pretrained_embeddings(vocab, emb_dim=300):
    print("Loading FastText embeddings (this might take a while)...")
    try:
        # 'fasttext-wiki-news-subwords-300' hem kelime hem alt kelime bilgisi içerir
        fasttext = api.load('fasttext-wiki-news-subwords-300') 
        embedding_matrix = torch.zeros(vocab.n_words, emb_dim)
        hits = 0
        misses = 0
        
        for i, word in vocab.index2word.items():
            if i < 4: continue # Skip special tokens
            try:
                embedding_matrix[i] = torch.tensor(fasttext[word])
                hits += 1
            except KeyError:
                # OOV words için random initialization (veya UNK vector)
                embedding_matrix[i] = torch.randn(emb_dim)
                misses += 1
        
        print(f"Embedding Loaded. Hits: {hits}, Misses: {misses}")
        return embedding_matrix
    except Exception as e:
        print(f"Failed to load FastText: {e}")
        return None

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
        src_indices = [self.src_vocab.word2index["<SOS>"]] + self.src_vocab.sentence_to_indices(src_text) + [self.src_vocab.word2index["<EOS>"]]
        trg_indices = [self.trg_vocab.word2index["<SOS>"]] + self.trg_vocab.sentence_to_indices(trg_text) + [self.trg_vocab.word2index["<EOS>"]]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# ==========================================
# 3. Model: LSTM with Pretrained Embeddings
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            # self.embedding.weight.requires_grad = False # Freeze etmek isterseniz açın
            
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch, seq_len]
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        # Target language (TR) için de fasttext yüklenebilir ama demo için random init yapıyoruz
        # (veya EN-TR çok dilli embedding kullanılabilir)
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
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src)
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1) 
            input = trg[:, t] if np.random.random() < teacher_forcing_ratio else top1
            
        return outputs

# ==========================================
# 4. Eğitim & Test
# ==========================================

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 100 == 0: print(f"Batch {i} loss: {loss.item():.4f}")
    return epoch_loss / len(iterator)

def evaluate_model_metrics(model, iterator, src_vocab, trg_vocab):
    model.eval()
    metric_bleu = evaluate.load("sacrebleu")
    metric_meteor = evaluate.load("meteor")
    metric_bert = evaluate.load("bertscore")
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            batch_size = src.shape[0]
            hidden, cell = model.encoder(src)
            input = torch.tensor([trg_vocab.word2index["<SOS>"]], device=DEVICE).repeat(batch_size)
            
            # Simple greedy decoding
            decoded_batch = []
            for _ in range(MAX_LEN):
                output, hidden, cell = model.decoder(input, hidden, cell)
                top1 = output.argmax(1)
                input = top1
                decoded_batch.append(top1.cpu().numpy())
            decoded_batch = np.array(decoded_batch).T # [batch, len]
            
            trg_indices = trg.cpu().numpy()
            for j in range(batch_size):
                pred_words = [trg_vocab.index2word[idx] for idx in decoded_batch[j] if idx not in [0, 1, 2]]
                trg_words = [trg_vocab.index2word[idx] for idx in trg_indices[j] if idx not in [0, 1, 2]]
                hypotheses.append(" ".join(pred_words))
                references.append([" ".join(trg_words)])
            if len(hypotheses) > 100: break

    print("Computing Metrics...")
    result_bleu = metric_bleu.compute(predictions=hypotheses, references=references)
    try: result_meteor = metric_meteor.compute(predictions=hypotheses, references=references)
    except: result_meteor = {"meteor": 0.0}
    result_bert = metric_bert.compute(predictions=hypotheses, references=[r[0] for r in references], lang="tr")
    
    print(f"BLEU: {result_bleu['score']:.2f}")
    print(f"METEOR: {result_meteor['meteor']:.4f}")
    print(f"BERTScore F1: {np.mean(result_bert['f1']):.4f}")


def main():
    print("Loading Dataset...")
    try:
        raw_dataset = load_from_disk(DATASET_PATH)
    except:
        # Fallback if cached deleted
        from datasets import load_dataset
        raw_dataset = load_dataset("open_subtitles", lang1="en", lang2="tr", trust_remote_code=True)
        raw_dataset.save_to_disk(DATASET_PATH)

    train_data = raw_dataset['train']
    SAMPLE_SIZE = 50000
    if len(train_data) > SAMPLE_SIZE: train_data = train_data.select(range(SAMPLE_SIZE))

    print("Building Vocabularies...")
    src_vocab = Vocabulary(SRC_LANG)
    trg_vocab = Vocabulary(TRG_LANG)
    for item in train_data:
        src_vocab.add_sentence(item['translation'][SRC_LANG])
        trg_vocab.add_sentence(item['translation'][TRG_LANG])
    src_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    trg_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    
    dataset = TranslationDataset(train_data, src_vocab, trg_vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Pretrained Load
    pretrained_emb = load_pretrained_embeddings(src_vocab, EMBED_DIM)
    
    print("Initializing LSTM Model...")
    enc = Encoder(src_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, pretrained_emb)
    dec = Decoder(trg_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Starting Training...")
    for epoch in range(N_EPOCHS):
        loss = train(model, dataloader, optimizer, criterion, CLIP)
        print(f"Epoch {epoch+1} Loss: {loss:.3f} | PPL: {np.exp(loss):.3f}")
        
    torch.save(model.state_dict(), 'fasttext_lstm_model.pt')
    print("Evaluating...")
    evaluate_model_metrics(model, dataloader, src_vocab, trg_vocab)

if __name__ == "__main__":
    main()
