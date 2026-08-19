import math
import torch
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tokenizer import *
from embedding import *
from core import *
from misc import *
def make_output(d_model, target_vocab_size):
    return nn.Linear(d_model, target_vocab_size)

def scheduler_(step, d_model, warmup_steps=4000):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

def train_one_batch(batch, encoder, decoder, output_layer, optimizer, loss_fn, device, accum_steps):
    encoder_input = batch["encoder_input"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    label = batch["label"].to(device)
    src_padding_mask = batch["src_padding_mask"].to(device)
    target_padding_mask = batch["target_padding_mask"].to(device)
    casual_mask = batch["casual_mask"].to(device)

    src_mask = src_padding_mask.unsqueeze(1).unsqueeze(1)
    target_mask = target_padding_mask.unsqueeze(1).unsqueeze(1) & casual_mask

    # NOTE: zero_grad() and step() REMOVED from here — they now live in main.py's loop

    encoder_output = encoder(encoder_input, src_mask)
    decoder_output, _ = decoder(decoder_input, encoder_output, target_mask, src_mask)
    logits = output_layer(decoder_output)

    loss = loss_fn(logits.reshape(-1, logits.size(-1)), label.reshape(-1))
    preds = logits.argmax(dim=-1)

    pad_id = loss_fn.ignore_index
    mask = label != pad_id

    correct = ((preds == label) & mask).sum().item()
    total = mask.sum().item()

    (loss / accum_steps).backward()

    return loss.item(), correct, total


@torch.no_grad()
def evaluate(loader, encoder, decoder, output_layer, loss_fn, device):
    """Average loss over a loader with no gradient updates -- used for
    val loss during training and could also be reused for test loss."""
    encoder.eval()
    decoder.eval()
    output_layer.eval()

    total_loss = 0.0
    num_batches = 0
    for batch in loader:
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
        total_loss += loss.item()
        num_batches += 1

    encoder.train()
    decoder.train()
    output_layer.train()

    return total_loss / num_batches