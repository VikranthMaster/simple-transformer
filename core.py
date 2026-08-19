"""
core.py

Transformer (encoder/decoder) implementation + dataset/collate helpers +
greedy/beam decoding, sized and stabilized for training on a 4GB GPU
(e.g. GTX 1650) without Tensor Cores.

Key fixes vs. a "textbook" implementation, aimed at the NaN-loss issue:

1. Pre-norm residual blocks (LayerNorm BEFORE each sub-layer, not after).
   Pre-norm keeps gradients much better behaved early in training and is
   the single biggest lever against NaN loss in from-scratch Transformers.

2. Fully-masked-row guard in attention. If every key for a given query
   position is masked out (can happen with padding), the raw scores row
   is all -inf and softmax(-inf, -inf, ...) = NaN, which then poisons
   the whole batch via backprop. We detect that and zero those rows out
   before softmax so they resolve to a harmless uniform distribution
   instead of NaN. The output there is discarded downstream anyway
   (it's a padding position), so this has no effect on real predictions.

3. No torch.cuda.amp autocast / GradScaler used anywhere. The GTX 1650
   (TU117) has NO Tensor Cores, so fp16 autocast buys you zero speedup
   on this GPU and only adds fp16 overflow risk (attention logits can
   exceed fp16's ~65504 max). Training in plain fp32 removes an entire
   class of NaN causes with no real downside on this card.
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (B, T, d_model)
        mask:    (B, 1, T_q, T_k) or broadcastable, bool, True = attend, False = ignore
        """
        B = q.size(0)

        q = self.w_q(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

            # --- NaN guard: fully-masked rows -------------------------------
            # If a query position has literally no valid key to attend to
            # (all False along the last dim), the row is all -inf and
            # softmax would produce NaN. Zero those rows so softmax gives a
            # harmless uniform distribution instead. The corresponding
            # output positions are padding and get ignored by the loss
            # mask / downstream slicing anyway.
            fully_masked = (mask == 0).all(dim=-1, keepdim=True)
            scores = torch.where(fully_masked, torch.zeros_like(scores), scores)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(out), attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Encoder / Decoder layers (pre-norm)
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask, cross_mask):
        normed = self.norm1(x)
        self_attn_out, _ = self.self_attn(normed, normed, normed, self_mask)
        x = x + self.dropout(self_attn_out)

        normed = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(normed, enc_out, enc_out, cross_mask)
        x = x + self.dropout(cross_attn_out)

        normed = self.norm3(x)
        ff_out = self.ff(normed)
        x = x + self.dropout(ff_out)
        return x, cross_attn_weights


# ---------------------------------------------------------------------------
# Encoder / Decoder stacks
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff=None, max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff=None, max_len=512, dropout=0.1, pad_id=0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, self_mask, cross_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, enc_out, self_mask, cross_mask)
        return self.norm(x), attn_weights


def make_output(d_model, target_vocab_size):
    return nn.Linear(d_model, target_vocab_size)


# ---------------------------------------------------------------------------
# LR schedule (Transformer "noam" schedule)
# ---------------------------------------------------------------------------

def scheduler_(step, d_model, warmup_steps=4000):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_pairs(df, tok, src_col="en", tgt_col="de"):
    """
    Turns a dataframe of (src, tgt) sentence pairs into tokenized tensors:
      encoder_input = src_tokens + [</s>]
      decoder_input = [<s>] + tgt_tokens          (teacher forcing input)
      label         = tgt_tokens + [</s>]         (shifted target for loss)
    """
    sos_id = tok.token_to_id("<s>")
    eos_id = tok.token_to_id("</s>")

    examples = []
    for src, tgt in tqdm(zip(df[src_col], df[tgt_col]), total=len(df), desc="Tokenizing pairs"):
        src_ids = tok.encode(src).ids
        tgt_ids = tok.encode(tgt).ids

        encoder_input = src_ids + [eos_id]
        decoder_input = [sos_id] + tgt_ids
        label = tgt_ids + [eos_id]

        examples.append({
            "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        })
    return examples


def make_collate(pad_id):
    def collate_fn(batch):
        encoder_inputs = [item["encoder_input"] for item in batch]
        decoder_inputs = [item["decoder_input"] for item in batch]
        labels = [item["label"] for item in batch]

        encoder_input = pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_id)
        decoder_input = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_id)
        label = pad_sequence(labels, batch_first=True, padding_value=pad_id)

        # True = real token, False = padding
        src_padding_mask = encoder_input != pad_id
        target_padding_mask = decoder_input != pad_id

        tgt_len = decoder_input.size(1)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_padding_mask": src_padding_mask,
            "target_padding_mask": target_padding_mask,
            "casual_mask": causal_mask,  # kept as "casual_mask" to match main.py
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Train / eval step
# ---------------------------------------------------------------------------

def train_one_batch(batch, encoder, decoder, output_layer, optimizer, loss_fn,
                     device, accum_steps):
    encoder_input = batch["encoder_input"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    label = batch["label"].to(device)
    src_padding_mask = batch["src_padding_mask"].to(device)
    target_padding_mask = batch["target_padding_mask"].to(device)
    casual_mask = batch["casual_mask"].to(device)

    src_mask = src_padding_mask.unsqueeze(1).unsqueeze(1)
    target_mask = target_padding_mask.unsqueeze(1).unsqueeze(1) & casual_mask

    encoder_output = encoder(encoder_input, src_mask)
    decoder_output, _ = decoder(decoder_input, encoder_output, target_mask, src_mask)
    logits = output_layer(decoder_output)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.reshape(-1))

    # Guard: if the forward pass itself produced NaN/inf, skip this batch
    # entirely instead of poisoning the running loss total / calling
    # backward() on garbage.
    if not torch.isfinite(loss):
        print("Warning: non-finite loss encountered, skipping this batch.")
        optimizer.zero_grad()
        return 0.0, 0, 0

    preds = logits.argmax(dim=-1)
    pad_id = loss_fn.ignore_index
    mask = label != pad_id
    correct = ((preds == label) & mask).sum().item()
    total = mask.sum().item()

    (loss / accum_steps).backward()
    return loss.item(), correct, total


@torch.no_grad()
def evaluate(loader, encoder, decoder, output_layer, loss_fn, device):
    encoder.eval(); decoder.eval(); output_layer.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        encoder_input = batch["encoder_input"].to(device, non_blocking=True)
        decoder_input = batch["decoder_input"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        src_padding_mask = batch["src_padding_mask"].to(device, non_blocking=True)
        target_padding_mask = batch["target_padding_mask"].to(device).bool()
        casual_mask = batch["casual_mask"].to(device).bool()

        src_mask = src_padding_mask.bool().unsqueeze(1).unsqueeze(1)
        target_mask = target_padding_mask.unsqueeze(1).unsqueeze(1) & casual_mask

        encoder_output = encoder(encoder_input, src_mask)
        decoder_output, _ = decoder(decoder_input, encoder_output, target_mask, src_mask)
        logits = output_layer(decoder_output)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.reshape(-1))

        if torch.isfinite(loss):
            total_loss += loss.item()
            num_batches += 1

    encoder.train(); decoder.train(); output_layer.train()
    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Inference: greedy + beam search decoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(src_ids, encoder, decoder, output_layer, tok, max_len, device):
    encoder.eval(); decoder.eval(); output_layer.eval()

    sos_id = tok.token_to_id("<s>")
    eos_id = tok.token_to_id("</s>")

    src_ids = src_ids.to(device)
    src_mask = torch.ones_like(src_ids, dtype=torch.bool).unsqueeze(1).unsqueeze(1)
    enc_out = encoder(src_ids, src_mask)

    ys = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        tgt_len = ys.size(1)
        causal = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device))
        tgt_mask = causal.unsqueeze(0).unsqueeze(0)

        out, _ = decoder(ys, enc_out, tgt_mask, src_mask)
        logits = output_layer(out[:, -1])
        next_id = logits.argmax(-1).item()

        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id:
            break

    encoder.train(); decoder.train(); output_layer.train()
    return ys.squeeze(0)


@torch.no_grad()
def beam_search_decode(src_ids, encoder, decoder, output_layer, tok, max_len, device, beam_width=4):
    encoder.eval(); decoder.eval(); output_layer.eval()

    sos_id = tok.token_to_id("<s>")
    eos_id = tok.token_to_id("</s>")

    src_ids = src_ids.to(device)
    src_mask = torch.ones_like(src_ids, dtype=torch.bool).unsqueeze(1).unsqueeze(1)
    enc_out = encoder(src_ids, src_mask)

    beams = [(torch.tensor([[sos_id]], dtype=torch.long, device=device), 0.0)]
    completed = []

    for _ in range(max_len - 1):
        candidates = []
        for ys, score in beams:
            if ys[0, -1].item() == eos_id:
                completed.append((ys, score))
                continue

            tgt_len = ys.size(1)
            causal = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device))
            tgt_mask = causal.unsqueeze(0).unsqueeze(0)

            out, _ = decoder(ys, enc_out, tgt_mask, src_mask)
            logits = output_layer(out[:, -1])
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            topk_probs, topk_ids = log_probs.topk(beam_width)
            for i in range(beam_width):
                next_id = topk_ids[i].item()
                next_score = score + topk_probs[i].item()
                next_ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
                candidates.append((next_ys, next_score))

        if not candidates:
            break

        # length-normalized score so beam search doesn't just prefer short sequences
        candidates.sort(key=lambda c: c[1] / c[0].size(1), reverse=True)
        beams = candidates[:beam_width]

        if all(b[0][0, -1].item() == eos_id for b in beams):
            completed.extend(beams)
            break

    completed.extend(beams)
    completed.sort(key=lambda c: c[1] / c[0].size(1), reverse=True)
    best = completed[0][0] if completed else beams[0][0]

    encoder.train(); decoder.train(); output_layer.train()
    return best.squeeze(0)
