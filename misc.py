import math
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from tokenizer import * 
from embedding import *
from core import *

def make_pairs(df, tokenizer):
    bos_id = tokenizer.token_to_id("<s>") 
    eos_id = tokenizer.token_to_id("</s>") 

    examples = []
    
    # We wrap zip(...) with tqdm(...) and provide the total length for the progress bar
    for en, de in tqdm(zip(df["en"], df["de"]), total=len(df), desc="Tokenizing Data"): 
        src_ids = tokenizer.encode(en).ids
        target_ids = [bos_id] + tokenizer.encode(de).ids + [eos_id]

        examples.append({
            "encoder_input": src_ids, 
            "decoder_input": target_ids[:-1], 
            "label": target_ids[1:]
        })

    return examples

class TranslationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ex = self.examples[index]
        return (
            torch.tensor(ex["encoder_input"], dtype=torch.long),
            torch.tensor(ex["decoder_input"], dtype=torch.long),
            torch.tensor(ex["label"], dtype=torch.long)
        )

def make_collate(pad_id):
    """padding the sequence"""
    def collate_fn(batch):
        enc_input, dec_input, label = zip(*batch)
        enc_input = pad_sequence(enc_input, batch_first=True, padding_value=pad_id)
        dec_input = pad_sequence(dec_input, batch_first=True, padding_value=pad_id)
        label = pad_sequence(label, batch_first=True, padding_value=pad_id)

        src_padding_mask = (enc_input != pad_id)
        target_padding_mask = (dec_input != pad_id)

        # masking
        target_len = dec_input.size(1)
        casual_mask = torch.tril(torch.ones(target_len, target_len, dtype=torch.bool))

        return {
            "encoder_input": enc_input,
            "decoder_input": dec_input,
            "label": label,
            "src_padding_mask": src_padding_mask,
            "target_padding_mask": target_padding_mask,
            "casual_mask": casual_mask
        }

    return collate_fn

def split_dataset(df, val_size, test_size, seed):
    """Split into train/val/test BEFORE tokenizer fitting or length
    filtering, so nothing about val/test leaks into training decisions."""
    train_df, temp_df = train_test_split(df, test_size=val_size + test_size, random_state=seed)
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, random_state=seed)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


@torch.no_grad()
def greedy_decode(src_ids, encoder, decoder, output_layer, tokenizer, max_len, device):
    """Generate a translation one token at a time, always picking the
    single highest-probability next token (no exploration of alternatives).
    Fast and simple, but can get stuck in locally-good-but-globally-bad
    choices since it never reconsiders earlier picks.
 
    src_ids: (1, src_len) -- a single sentence's encoder input ids
    """
    encoder.eval(); decoder.eval(); output_layer.eval()
 
    bos_id = tokenizer.token_to_id("<s>") # Changed to parentheses
    eos_id = tokenizer.token_to_id("</s>") # Changed to parentheses
 
    src_ids = src_ids.to(device)
    src_mask = None  # no padding to mask for a single un-batched sentence
 
    encoder_output = encoder(src_ids, src_mask)
 
    decoder_input = torch.tensor([[bos_id]], dtype=torch.long, device=device)
 
    for _ in range(max_len):
        tgt_len = decoder_input.size(1)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device))
        tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)
 
        decoder_output, _ = decoder(decoder_input, encoder_output, tgt_mask, src_mask)
        logits = output_layer(decoder_output)         # (1, tgt_len, vocab_size)
        next_token_logits = logits[0, -1, :]           # only care about the newest position
        next_token = torch.argmax(next_token_logits).item()
 
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], device=device)], dim=1
        )
 
        if next_token == eos_id:
            break
 
    output_ids = decoder_input[0].tolist()
    return output_ids
 
 
@torch.no_grad()
def beam_search_decode(src_ids, encoder, decoder, output_layer, tokenizer, max_len, device, beam_width=4):
    """Keeps the top `beam_width` candidate sequences at every step
    (instead of just the single best one), so it can recover from a
    token choice that looks good short-term but leads somewhere worse.
    Slower than greedy (beam_width x the compute), usually better quality.
    """
    encoder.eval(); decoder.eval(); output_layer.eval()
 
    bos_id = tokenizer.token_to_id("<s>") # Changed to parentheses
    eos_id = tokenizer.token_to_id("</s>") # Changed to parentheses
 
    src_ids = src_ids.to(device)
    src_mask = None
    encoder_output = encoder(src_ids, src_mask)
 
    # Each beam: (token_id_list, cumulative_log_prob, finished_flag)
    beams = [([bos_id], 0.0, False)]
 
    for _ in range(max_len):
        all_candidates = []
 
        for tokens, score, finished in beams:
            if finished:
                all_candidates.append((tokens, score, finished))
                continue
 
            decoder_input = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_len = decoder_input.size(1)
            causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device))
            tgt_mask = causal_mask.unsqueeze(0).unsqueeze(0)
 
            decoder_output, _ = decoder(decoder_input, encoder_output, tgt_mask, src_mask)
            logits = output_layer(decoder_output)
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)   # (vocab_size,)
 
            top_log_probs, top_ids = torch.topk(log_probs, beam_width)
 
            for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                new_tokens = tokens + [token_id]
                new_score = score + log_prob
                new_finished = (token_id == eos_id)
                all_candidates.append((new_tokens, new_score, new_finished))
 
        # Length-normalized score so beams don't get unfairly penalized
        # just for being longer (log-probs are always negative, so
        # unnormalized scores would always favor shorter sequences)
        def normalized_score(candidate):
            tokens, score, _ = candidate
            return score / len(tokens)
 
        beams = sorted(all_candidates, key=normalized_score, reverse=True)[:beam_width]
 
        if all(finished for _, _, finished in beams):
            break
 
    best_tokens, _, _ = beams[0]
    return best_tokens
 
 
def ids_to_sentence(ids, tokenizer):
    """Convert generated ids back to readable text, dropping special tokens
    (<s>, </s>, <pad>) so they don't leak into the output as literal text."""
    
    return tokenizer.decode(ids, skip_special_tokens=True)
