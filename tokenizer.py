import json
import os

class Tokenizer:
    END_OF_WORD = "</w>"

    def __init__(self):
        self.merges = {}      
        self.merge_order = {} 
        self.vocab = set()
        self.token_to_id = {}
        self.id_to_token = {}

    def _word_to_symbols(self, word):
        return list(word) + [self.END_OF_WORD]

    def get_pair_counts(self, word_freqs):
        counts = {}
        for word_symbols, freq in word_freqs.items():
            for x, y in zip(word_symbols, word_symbols[1:]):
                counts[(x, y)] = counts.get((x, y), 0) + freq
        return counts

    def merge(self, word_symbols, pair, merged_symbol):
        new_symbols = []
        i = 0
        while i < len(word_symbols):
            if (i < len(word_symbols) - 1
                    and word_symbols[i] == pair[0]
                    and word_symbols[i + 1] == pair[1]):
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(word_symbols[i])
                i += 1
        return new_symbols

    def save_checkpoint(self, path, current_step, word_freqs):
        """Saves the intermediate training state so it can be resumed."""
        data = {
            "current_step": current_step,
            # JSON keys must be strings, so we convert the tuple keys to lists for storage
            "word_freqs": [[list(k), v] for k, v in word_freqs.items()],
            "merges": [[a, b] for a, b in sorted(self.merge_order, key=self.merge_order.get)],
        }
        # Write to a temporary file first, then rename to prevent corruption if power 
        # dies exactly while writing the checkpoint file.
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(temp_path, path)

    def load_checkpoint(self, path):
        """Loads the intermediate training state."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Restore merges
        self.merges = {}
        self.merge_order = {}
        for rank, (a, b) in enumerate(data["merges"]):
            self.merges[(a, b)] = a + b
            self.merge_order[(a, b)] = rank
            
        # Restore word_freqs (converting lists back to tuples)
        word_freqs = {tuple(k): v for k, v in data["word_freqs"]}
        return data["current_step"], word_freqs

    def bpe(self, text, num_merges, printing=False, checkpoint_path="bpe_checkpoint.json", save_every=100):
        """Train merges on a corpus, saving checkpoints periodically."""
        
        # Check if we are resuming from a previous run
        if os.path.exists(checkpoint_path):
            current_step, word_freqs = self.load_checkpoint(checkpoint_path)
            if printing:
                print(f"Resuming BPE training from step {current_step}...")
        else:
            current_step = 0
            if isinstance(text, str):
                lines = [text]
            else:
                lines = text

            # 1. Build word frequency table: {('t','h','e','</w>'): count, ...}
            word_freqs = {}
            for line in lines:
                for word in line.strip().split():
                    symbols = tuple(self._word_to_symbols(word))
                    word_freqs[symbols] = word_freqs.get(symbols, 0) + 1

        # 2. Greedily merge the most frequent pair
        for i in range(current_step, num_merges):
            pair_counts = self.get_pair_counts(word_freqs)
            if not pair_counts:
                break

            top_pair = max(pair_counts, key=pair_counts.get)
            merged_symbol = top_pair[0] + top_pair[1]

            new_word_freqs = {}
            for word_symbols, freq in word_freqs.items():
                new_symbols = tuple(self.merge(list(word_symbols), top_pair, merged_symbol))
                new_word_freqs[new_symbols] = new_word_freqs.get(new_symbols, 0) + freq
            word_freqs = new_word_freqs

            self.merges[top_pair] = merged_symbol
            self.merge_order[top_pair] = i
            if printing:
                print(f"{i}: Merged: {top_pair} -> '{merged_symbol}' (count={pair_counts[top_pair]})")

            # Save checkpoint periodically
            if (i + 1) % save_every == 0:
                self.save_checkpoint(checkpoint_path, i + 1, word_freqs)
                if printing:
                    print(f"--- Checkpoint saved at step {i + 1} ---")

        # 3. Build final vocab (all symbols seen across all words)
        for word_symbols in word_freqs:
            self.vocab.update(word_symbols)

        self.build_vocab()
        
        # Optional: Remove checkpoint file after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    def _encode_word(self, word):
        symbols = self._word_to_symbols(word)
        if len(symbols) == 1:
            return symbols

        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate = min(
                (p for p in pairs if p in self.merges),
                key=lambda p: self.merge_order[p],
                default=None,
            )
            if candidate is None:
                break
            merged_symbol = self.merges[candidate]
            symbols = self.merge(symbols, candidate, merged_symbol)

        return symbols

    def encode(self, text):
        tokens = []
        for word in text.strip().split():
            tokens.extend(self._encode_word(word))
        return tokens

    def decode(self, tokens):
        text = "".join(tokens)
        text = text.replace(self.END_OF_WORD, " ")
        return text.strip()

    def build_vocab(self):
        specials = ["<pad>","<s>","</s>","<unk>"]
        all_tokens = specials + sorted(self.vocab)
        self.token_to_id = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
    
    def tokens_to_ids(self, tokens):
        unk_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(tok, unk_id) for tok in tokens]

    def ids_to_tokens(self, ids):
        return [self.id_to_token.get(i, "<unk>") for i in ids]
    
    def save(self, path):
        data = {
            "merges": [[a, b] for a, b in sorted(self.merge_order, key=self.merge_order.get)],
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved final tokenizer to: {path}")
 
    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
 
        self.merges = {}
        self.merge_order = {}
        for rank, (a, b) in enumerate(data["merges"]):
            self.merges[(a, b)] = a + b
            self.merge_order[(a, b)] = rank
 
        self.token_to_id = data["token_to_id"]
        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        self.vocab = set(self.token_to_id.keys())