
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
import math

# ==========================================
# 1. Konfigürasyon
# ==========================================
DATASET_PATH = "open_subtitles_en_tr"
SRC_LANG = 'en'
TRG_LANG = 'tr'
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
EMBED_DIM = 256
HIDDEN_DIM = 256 # Kernel sayısı
KERNEL_SIZE = 3  # Convolution pencere boyutu
N_LAYERS = 5     # Derin Convolutional Encoder/Decoder
DROPOUT = 0.25
LEARNING_RATE = 0.0001 # CNN'ler için biraz daha düşük LR iyidir
N_EPOCHS = 10
CLIP = 0.1 # CNN'de gradient clipping daha hassas olabilir
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 50

# ==========================================
# 2. Veri Hazırlama (Aynı Yapı)
# ==========================================
# ... (Vocab ve Dataset sınıfları, tutarlılık için aynı tutuluyor)

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
        src_indices = [self.src_vocab.word2index["<SOS>"]] + self.src_vocab.sentence_to_indices(src_text) + [self.src_vocab.word2index["<EOS>"]]
        trg_indices = [self.trg_vocab.word2index["<SOS>"]] + self.trg_vocab.sentence_to_indices(trg_text) + [self.trg_vocab.word2index["<EOS>"]]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(trg_indices, dtype=torch.long)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    return src_batch, trg_batch

# ==========================================
# 3. Model Tanımları: Fully Convolutional Seq2Seq (ConvS2S)
# ==========================================
# Kaynak: "Convolutional Sequence to Sequence Learning" (Gehring et al., 2017)
# Bu model RNN veya Transformer kullanmaz. Tamamen 1D Convolution (Conv1d) kullanır.

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, device, max_length=100):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        # Convolution blokları (Conv1d)
        self.convs = nn.ModuleList([nn.Conv1d(
                                        in_channels=hid_dim, 
                                        out_channels=2 * hid_dim, # GLU için input'u ikiye katla
                                        kernel_size=kernel_size, 
                                        padding=(kernel_size - 1) // 2) # Padding ile boyut koruma (same)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Positional Embedding (Learned)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        # Embedding Toplama
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout(tok_embedded + pos_embedded) # [batch_size, src_len, emb_dim]
        
        # Conv input formatı: [batch_size, channels, length]
        conv_input = self.emb2hid(embedded).permute(0, 2, 1) # [batch_size, hid_dim, src_len]
        
        for i, conv in enumerate(self.convs):
            # Residual bağlantı için input'u sakla
            conved = conv(self.dropout(conv_input))
            # GLU (Gated Linear Unit) - 2*hid_dim boyutunu hid_dim'e düşürür ve gate uygular
            conved = nn.functional.glu(conved, dim=1)
            # Residual connection
            conved = (conved + conv_input) * self.scale
            conv_input = conved
            
        # Encoder çıktısı: [batch_size, src_len, emb_dim]    
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        # Combine with original embedding (Residual)
        combined = (conved + embedded) * self.scale
        
        return conved, combined

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, kernel_size, dropout, trg_pad_idx, device, max_length=100):
        super().__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(
                                        in_channels=hid_dim, 
                                        out_channels=2 * hid_dim, 
                                        kernel_size=kernel_size) 
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # Multi-step Attention
        # conved: [batch_size, hid_dim, trg_len]
        # encoder_conved: [batch_size, src_len, emb_dim]
        # encoder_combined: [batch_size, src_len, emb_dim]
        
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        # conved_emb: [batch_size, trg_len, emb_dim]
        
        combined = (conved_emb + embedded) * self.scale
        # combined: [batch_size, trg_len, emb_dim]
        
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        # energy: [batch_size, trg_len, src_len]
        
        attention = nn.functional.softmax(energy, dim=2)
        
        attention_encoding = torch.matmul(attention, encoder_combined)
        # [batch_size, trg_len, emb_dim]
        
        attention_encoding = self.attn_emb2hid(attention_encoding)
        # [batch_size, trg_len, hid_dim]
        
        attention_encoding = attention_encoding.permute(0, 2, 1)
        # [batch_size, hid_dim, trg_len]
        
        conved = (conved + attention_encoding) * self.scale
        return conved, attention

    def forward(self, trg, encoder_conved, encoder_combined):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        conv_input = self.emb2hid(embedded).permute(0, 2, 1)
        
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)
            
            # ConvS2S Decoder'da padding stratejisi kritiktir.
            # Geleceği görmemek için sadece sol tarafa padding eklenir.
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            padded_conv_input = torch.cat((padding, conv_input), dim=2)
            
            conved = conv(padded_conv_input)
            conved = nn.functional.glu(conved, dim=1)
            
            # Attention Mechanism
            conved, attention = self.calculate_attention(embedded, conved, encoder_conved, encoder_combined)
            
            conved = (conved + conv_input) * self.scale
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        output = self.fc_out(self.dropout(conved))
        return output

class ConvSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        encoder_conved, encoder_combined = self.encoder(src)
        output = self.decoder(trg, encoder_conved, encoder_combined)
        return output

# ==========================================
# 4. Eğitim ve Değerlendirme
# ==========================================
# Gerekli NLTK data
import nltk
try:
    nltk.data.find("wordnet")
except LookupError:
    nltk.download("wordnet")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        # ConvS2S output trg_len ile aynı boyuttadır çünkü padding ile input boyutu korunur
        # target input (trg_input) son token <EOS> hariç verilir
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]
        
        output = model(src, trg_input)
        # output: [batch_size, trg_len-1, output_dim]
        
        output_dim = output.shape[-1]
        output = output.reshape(-1, output_dim)
        trg_output = trg_output.reshape(-1)
        
        loss = criterion(output, trg_output)
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
            # CNN Based generation (basit greedy)
            # CNN'de output parallel üretilebilir ama generate adım adım yapılırsa dikkatli olunmalı.
            # Şimdilik model outputunu alıp argmax yapacağız (bu teach forcing olmadan tam autoregressive değil ama evaluation için hızlı bir yaklaşım)
            # Tam Autoregressive için loop içinde her adımda modele verilen trg büyütülmeli.
            
            # Autoregressive generation loop
            batch_size = src.shape[0]
            max_len = 50
            trg_input = torch.tensor([trg_vocab.word2index["<SOS>"]], device=DEVICE).repeat(batch_size, 1) # [batch, 1]
            
            for _ in range(max_len):
                output = model(src, trg_input) # [batch, seq_len, vocab]
                pred_token = output[:, -1].argmax(1).unsqueeze(1) # Son token tahmini [batch, 1]
                trg_input = torch.cat((trg_input, pred_token), dim=1)
                
            pred_indices = trg_input.cpu().numpy()
            trg_indices = trg.cpu().numpy()
            
            for j in range(batch_size):
                pred_words = [trg_vocab.index2word[idx] for idx in pred_indices[j] if idx not in [0, 1, 2]]
                trg_words = [trg_vocab.index2word[idx] for idx in trg_indices[j] if idx not in [0, 1, 2]]
                hypotheses.append(" ".join(pred_words))
                references.append([" ".join(trg_words)])
            
            if len(hypotheses) > 100: break

    result_bleu = metric_bleu.compute(predictions=hypotheses, references=references)
    try: result_meteor = metric_meteor.compute(predictions=hypotheses, references=references)
    except: result_meteor = {"meteor": 0.0}
    result_bert = metric_bert.compute(predictions=hypotheses, references=[r[0] for r in references], lang="tr")
    
    print(f"BLEU: {result_bleu['score']:.2f}")
    print(f"METEOR: {result_meteor['meteor']:.4f}")
    print(f"BERTScore F1: {np.mean(result_bert['f1']):.4f}")

def main():
    print("Loading data...")
    raw_dataset = load_from_disk(DATASET_PATH)
    train_data = raw_dataset['train']
    SAMPLE_SIZE = 50000
    if len(train_data) > SAMPLE_SIZE: train_data = train_data.select(range(SAMPLE_SIZE))

    src_vocab = Vocabulary(SRC_LANG)
    trg_vocab = Vocabulary(TRG_LANG)
    for item in train_data:
        src_vocab.add_sentence(item['translation'][SRC_LANG])
        trg_vocab.add_sentence(item['translation'][TRG_LANG])
    src_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    trg_vocab.build_vocab(MAX_VOCAB_SIZE, MIN_FREQ)
    
    dataset = TranslationDataset(train_data, src_vocab, trg_vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print("Initializing Convolutional Seq2Seq Model...")
    enc = Encoder(src_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, KERNEL_SIZE, DROPOUT, DEVICE)
    dec = Decoder(trg_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, KERNEL_SIZE, DROPOUT, 0, DEVICE)
    model = ConvSeq2Seq(enc, dec).to(DEVICE)
    
    print(f'Model Parameters: {count_parameters(model):,}')
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(N_EPOCHS):
        loss = train(model, dataloader, optimizer, criterion, CLIP)
        print(f"Epoch {epoch+1} Loss: {loss:.3f} | PPL: {np.exp(loss):.3f}")
        
    torch.save(model.state_dict(), 'cnn_seq2seq_model.pt')
    print("Evaluating...")
    evaluate_model_metrics(model, dataloader, src_vocab, trg_vocab)

if __name__ == "__main__":
    main()
