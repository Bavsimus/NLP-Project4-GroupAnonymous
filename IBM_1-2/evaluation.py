# --- STEP 4: TRAINING AND EVALUATION ---

# Initialize and Train
model = IBMModel2(train_data)
model.train(iterations=5)

# Evaluate on Test Set
print("\n>>> Evaluating model on Test Set...")
references, hypotheses = model.translate(test_data)

# Calculate BLEU Scores
bleu_1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

print("="*50)
print(f"EVALUATION RESULTS (Data size: {DATA_LIMIT})")
print("="*50)
print(f"BLEU-1 Score: {bleu_1:.4f} (Measures lexical accuracy)")
print(f"BLEU-4 Score: {bleu_4:.4f} (Measures fluency/grammar)")
print("-" * 50)

print("\nSample Translations:")
for i in range(5):
    print(f"Ref : {' '.join(references[i][0])}")
    print(f"Pred: {' '.join(hypotheses[i])}")
    print("." * 30)

# --- STEP 5: VISUALIZATION (ALIGNMENT HEATMAP) ---

def plot_heatmap(model, en_sentence, tr_sentence):
    """Generates an alignment heatmap for a given sentence pair."""
    en_tokens = model._clean_text(en_sentence).split()
    tr_tokens = model._clean_text(tr_sentence).split()
    
    alignment_matrix = np.zeros((len(tr_tokens), len(en_tokens)))
    
    for i, tr_word in enumerate(tr_tokens):
        for j, en_word in enumerate(en_tokens):
            alignment_matrix[i, j] = model.t_probs[(tr_word, en_word)]

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(alignment_matrix, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(en_tokens)))
    ax.set_yticks(np.arange(len(tr_tokens)))
    ax.set_xticklabels(en_tokens, rotation=45)
    ax.set_yticklabels(tr_tokens)
    
    plt.xlabel("Source (English)")
    plt.ylabel("Target (Turkish)")
    plt.title("Word Alignment Probability Heatmap")
    plt.show()

print("\n>>> Generating Alignment Visualization...")
# Example sentence for visualization
example_en = "I prefer to live in a wooden house"
example_tr = "ahşap bir evde yaşamayı tercih ederim"

plot_heatmap(model, example_en, example_tr)