# --- STEP 3: DEFINING IBM MODELS ---

class IBMModel2:
    def __init__(self, data):
        """
        Initializes IBM Model 2. 
        It requires training IBM Model 1 first to initialize lexical probabilities.
        """
        self.corpus = []
        self.vocab_en = set()
        self.vocab_tr = set()
        self.t_probs = {}  # Lexical translation probabilities: t(f|e)
        self.a_probs = {}  # Alignment probabilities: a(i|j, l, m)
        
        self._preprocess(data)
        self._initialize_model1()

    def _clean_text(self, text):
        """Removes punctuation and converts text to lowercase."""
        return text.lower().translate(str.maketrans('', '', string.punctuation))

    def _preprocess(self, data):
        print("Preprocessing corpus for IBM Model 2...")
        for item in data:
            # Adding NULL token is crucial for words that have no counterpart
            en_tokens = ["NULL"] + self._clean_text(item['en']).split()
            tr_tokens = self._clean_text(item['tr']).split()
            
            if not en_tokens or not tr_tokens: continue
            
            self.corpus.append((en_tokens, tr_tokens))
            self.vocab_en.update(en_tokens)
            self.vocab_tr.update(tr_tokens)

    def _initialize_model1(self):
        """
        Trains IBM Model 1 (Lexical Translation Only) to initialize probabilities.
        """
        print(">>> Phase 1: Training IBM Model 1 (Initialization)...")
        # Uniform initialization
        initial_prob = 1.0 / len(self.vocab_tr)
        self.t_probs = defaultdict(lambda: initial_prob)
        
        # Fast Model 1 Training (5 Iterations)
        for i in range(5):
            count_tr_en = defaultdict(float)
            total_en = defaultdict(float)
            
            for en_tokens, tr_tokens in self.corpus:
                for tr_word in tr_tokens:
                    s_total = 0.0
                    for en_word in en_tokens:
                        s_total += self.t_probs[(tr_word, en_word)]
                    
                    for en_word in en_tokens:
                        p = self.t_probs[(tr_word, en_word)] / s_total
                        count_tr_en[(tr_word, en_word)] += p
                        total_en[en_word] += p
            
            # M-Step: Update probabilities
            for (tr_word, en_word), val in count_tr_en.items():
                self.t_probs[(tr_word, en_word)] = val / total_en[en_word]
        
        print("IBM Model 1 training complete.")
        # Initialize alignment probabilities uniformly
        self.a_probs = defaultdict(lambda: 1.0) 

    def train(self, iterations=5):
        """
        Trains IBM Model 2 (Adds Alignment/Position Knowledge).
        """
        print(f">>> Phase 2: Training IBM Model 2 ({iterations} iterations)...")
        
        for it in range(iterations):
            count_tr_en = defaultdict(float)
            total_en = defaultdict(float)
            count_align = defaultdict(float) 
            total_align = defaultdict(float) 
            
            for en_tokens, tr_tokens in self.corpus:
                l = len(en_tokens) - 1 
                m = len(tr_tokens)
                
                # E-Step (Expectation)
                for j, tr_word in enumerate(tr_tokens):
                    s_total = 0.0
                    for i, en_word in enumerate(en_tokens):
                        align_prob = self.a_probs.get((i, j, l, m), 1.0 / (l + 1))
                        s_total += self.t_probs[(tr_word, en_word)] * align_prob
                    
                    if s_total == 0: continue

                    for i, en_word in enumerate(en_tokens):
                        align_prob = self.a_probs.get((i, j, l, m), 1.0 / (l + 1))
                        p = (self.t_probs[(tr_word, en_word)] * align_prob) / s_total
                        
                        count_tr_en[(tr_word, en_word)] += p
                        total_en[en_word] += p
                        count_align[(i, j, l, m)] += p
                        total_align[(j, l, m)] += p
            
            # M-Step (Maximization)
            for (tr_word, en_word), val in count_tr_en.items():
                self.t_probs[(tr_word, en_word)] = val / total_en[en_word]
            
            self.a_probs.clear()
            for (i, j, l, m), val in count_align.items():
                self.a_probs[(i, j, l, m)] = val / total_align[(j, l, m)]
                
            print(f"Iteration {it + 1} complete.")

    def translate(self, test_set):
        """Translates sentences using the learned probabilities."""
        predictions = []
        references = []
        
        # Caching best translations for performance
        best_map = {}
        for (tr, en), prob in self.t_probs.items():
            if en not in best_map or prob > best_map[en][1]:
                best_map[en] = (tr, prob)
        
        for item in test_set:
            en_text = self._clean_text(item['en']).split()
            tr_ref = self._clean_text(item['tr']).split()
            
            pred_tokens = []
            for word in en_text:
                if word in best_map:
                    pred_tokens.append(best_map[word][0])
            
            if not pred_tokens: pred_tokens = ["."]
            predictions.append(pred_tokens)
            references.append([tr_ref])
            
        return references, predictions