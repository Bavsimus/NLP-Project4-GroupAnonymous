
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
import math

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
N_HEADS = 8
HIDDEN_DIM = 512 # FeedForward boyutu
N_LAYERS = 3
DROPOUT = 0.1
LEARNING_RATE = 0.0005
N_EPOCHS = 10
CLIP = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 50 # Transformer konumsal kodlama için

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
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# ==========================================
# 3. Model Tanımları (Transformer)
# ==========================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_len=100, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        
    def create_mask(self, src, trg):
        src_seq_len = src.shape[1]
        trg_seq_len = trg.shape[1]
        
        # Target look-ahead mask (causal mask)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(src.device)
        
        # Padding masks (True where padding exists)
        src_padding_mask = (src == self.pad_idx)
        trg_padding_mask = (trg == self.pad_idx)
        
        return trg_mask, src_padding_mask, trg_padding_mask

    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        trg_mask, src_padding_mask, trg_padding_mask = self.create_mask(src, trg)
        
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        trg_emb = self.pos_encoder(self.trg_embedding(trg) * math.sqrt(self.d_model))
        
        output = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
            memory_key_padding_mask=src_padding_mask # Encoder padding maskesi memory için de kullanılır
        )
        
        return self.fc_out(output)

# ==========================================
# 4. Eğitim ve Yardımcı Fonksiyonlar
# ==========================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        
        # Transformer'da target input (trg_input) son token hariç, 
        # target output (trg_output) ise ilk token (<SOS>) hariçtir.
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]
        
        optimizer.zero_grad()
        
        output = model(src, trg_input)
        # output: [batch_size, trg_len - 1, output_dim]
        
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg_output = trg_output.reshape(-1)
        
        loss = criterion(output, trg_output)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 100 == 0:
            print(f"Batch {i} loss: {loss.item():.4f}")
        
    return epoch_loss / len(iterator)


import evaluate # HuggingFace Evaluate
import nltk
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")

def evaluate_model(model, iterator, src_vocab, trg_vocab, max_len=50):
    model.eval()
    metric_bleu = evaluate.load("sacrebleu")
    metric_meteor = evaluate.load("meteor")
    metric_bert = evaluate.load("bertscore")
    
    hypotheses = []
    references = []
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            # trg: [batch_size, trg_len]
            
            # Greedy Decoding for evaluation
            batch_size = src.shape[0]
            
            # Encoder input prepare
            # forward expects: src, trg
            # For generation, we need to loop manually or write a generate method.
            # Simplified generate loop for Transformer:
            
            # Encode
            trg_input = torch.tensor([trg_vocab.word2index["<SOS>"]], device=DEVICE).repeat(batch_size, 1)
            
            # Basic generation loop (inefficient but works for metrics)
            # Note: For Transformer, we need to call model(src, trg_input) iteratively
            
            for _ in range(max_len):
                output = model(src, trg_input) # [batch, seq, vocab]
                next_token_logits = output[:, -1, :]
                next_token = next_token_logits.argmax(1).unsqueeze(1)
                trg_input = torch.cat((trg_input, next_token), dim=1)
                
                # Check for EOS? (Skipped for parallel efficiency in simple loop, usually handled by cutting at EOS later)
            
            pred_indices = trg_input.cpu().numpy()
            trg_indices = trg.cpu().numpy()
            
            for j in range(batch_size):
                pred_words = [trg_vocab.index2word[idx] for idx in pred_indices[j] if idx not in [0, 1, 2]] # Removing PAD, SOS, EOS
                trg_words = [trg_vocab.index2word[idx] for idx in trg_indices[j] if idx not in [0, 1, 2]]
                
                hypotheses.append(" ".join(pred_words))
                references.append([" ".join(trg_words)])
            
            # Limit evaluation for speed if needed
            if len(hypotheses) > 100: 
                break

    print("Computing metrics...")
    result_bleu = metric_bleu.compute(predictions=hypotheses, references=references)
    result_meteor = metric_meteor.compute(predictions=hypotheses, references=references)
    result_bert = metric_bert.compute(predictions=hypotheses, references=[r[0] for r in references], lang="tr")
    
    print(f"BLEU: {result_bleu['score']:.2f}")
    print(f"METEOR: {result_meteor['meteor']:.4f}")
    print(f"BERTScore F1: {np.mean(result_bert['f1']):.4f}")

    return result_bleu['score']

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        raw_dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"Hata: Dataset yüklenemedi. Lütfen önce Dataset.py dosyasını çalıştırıp dataseti indirin. Hata detayı: {e}")
        return

    train_data = raw_dataset['train']
    
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
    # Split for eval? reusing train for demo simplicity as per original code
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print("Initializing Transformer Model...")
    model = TransformerModel(
        src_vocab_size=src_vocab.n_words,
        trg_vocab_size=trg_vocab.n_words,
        d_model=EMBED_DIM,
        nhead=N_HEADS,
        num_encoder_layers=N_LAYERS,
        num_decoder_layers=N_LAYERS,
        dim_feedforward=HIDDEN_DIM,
        dropout=DROPOUT,
        max_len=100,
        pad_idx=0
    ).to(DEVICE)
    
    # Xavier initialization often helps Transformers
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # 0 is PAD
    
    print("Starting Training...")
    for epoch in range(N_EPOCHS):
        train_loss = train(model, dataloader, optimizer, criterion, CLIP)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | PPL: {np.exp(train_loss):.3f}')
        
    torch.save(model.state_dict(), 'transformer_seq2seq_model.pt')
    print("Model saved!")
    
    print("Evaluating Model...")
    evaluate_model(model, dataloader, src_vocab, trg_vocab)


if __name__ == "__main__":
    main()
