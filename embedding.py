import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# from tokenizer import *

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)


def positional_encoding(seq_len: int, d_model: int, device=None):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model)
    )
 
    pe[:, 0::2] = torch.sin(position * div_term)  
    pe[:, 1::2] = torch.cos(position * div_term)  
 
    return pe.unsqueeze(0)

def check_max_len(df, tokenizer, percentile: float = 99):
    """Computes sequence lengths across the whole dataset and reports
    both the true max and a percentile cutoff -- because a single long
    outlier sentence shouldn't force every batch to pad to a huge length.
    """
    # 1. FIXED: Added .ids to properly get the length of the Hugging Face Encoding
    # 2. FIXED: Changed "english" to "en" and "korean" to "de" to match your dataset
    en_lens = [len(tokenizer.encode(s).ids) for s in df["en"]]
    # +2 accounts for <s> and </s> added to the target side
    de_lens = [len(tokenizer.encode(s).ids) + 2 for s in df["de"]]
 
    all_lens = en_lens + de_lens
    all_lens_sorted = sorted(all_lens)
 
    true_max = max(all_lens)
    cutoff_idx = int(len(all_lens_sorted) * percentile / 100)
    percentile_len = all_lens_sorted[min(cutoff_idx, len(all_lens_sorted) - 1)]
 
    print(f"English: min={min(en_lens)}, max={max(en_lens)}, avg={sum(en_lens)/len(en_lens):.1f}")
    print(f"German:  min={min(de_lens)}, max={max(de_lens)}, avg={sum(de_lens)/len(de_lens):.1f}")
    print(f"\nTrue max across both sides: {true_max}")
    print(f"{percentile}th percentile:          {percentile_len}")
    print(f"\nRecommendation: use max_len ~= {percentile_len} "
          f"(round up a bit, e.g. to a multiple of 8), and either "
          f"truncate or drop the small number of sentences longer than this.")
 
    return true_max, percentile_len