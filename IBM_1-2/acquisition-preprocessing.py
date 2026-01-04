import os
import random
import string
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu

# --- STEP 1: DATA ACQUISITION ---
print(">>> Step 1: Downloading and extracting Tatoeba (English-Turkish) dataset...")

# Download dataset directly from source to avoid library dependency issues
if not os.path.exists("tatoeba_en_tr.zip"):
    !wget -q -O tatoeba_en_tr.zip https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/en-tr.txt.zip
    !unzip -o tatoeba_en_tr.zip

# Load data into memory
dataset = []
DATA_LIMIT = 30000  # Increased limit for better accuracy

try:
    with open("Tatoeba.en-tr.en", "r", encoding="utf-8") as f_en, \
         open("Tatoeba.en-tr.tr", "r", encoding="utf-8") as f_tr:
        
        for en_line, tr_line in zip(f_en, f_tr):
            dataset.append({
                'en': en_line.strip(),
                'tr': tr_line.strip()
            })
            if len(dataset) >= DATA_LIMIT:
                break
    print(f"Data loaded successfully. Total sentences: {len(dataset)}")

except FileNotFoundError:
    print("Error: Dataset files not found.")

# --- STEP 2: TRAIN / TEST SPLIT ---
# Splitting data: 90% for Training, 10% for Testing
random.seed(42)
random.shuffle(dataset)

split_idx = int(len(dataset) * 0.9)
train_data = dataset[:split_idx]
test_data = dataset[split_idx:]

print(f"Training Set: {len(train_data)} sentences")
print(f"Test Set    : {len(test_data)} sentences")