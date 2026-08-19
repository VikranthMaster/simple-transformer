"""
main.py

Training script for the EN->DE translation Transformer.

Changes vs. the original, and why:

1. Removed torch.cuda.amp autocast()/GradScaler entirely.
   The GTX 1650 has no Tensor Cores, so fp16 autocast gives zero speedup
   there and only adds fp16 overflow risk -- a common source of NaN loss.
   Plain fp32 training removes that risk with no real downside on this GPU.

2. Shrunk the default model (D_MODEL, NUM_LAYERS, D_FF) to comfortably fit
   in 4GB VRAM. The original sizing (512 / 6 layers) was tuned for a T4's
   16GB. Bump these back up if/when you train on a bigger card again --
   they're the ones to raise first.

3. Filled in the placeholder file paths (FIX 1 in your version) -- set
   TRAIN_PATH / VAL_PATH / TEST_PATH / TOKENIZER_PATH to your actual files
   below.

4. Gradient clipping now runs directly against the raw gradients (no
   scaler.unscale_ step needed, since there's no scaler).
"""

import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from core import (
    Encoder,
    Decoder,
    make_output,
    scheduler_,
    train_one_batch,
    evaluate,
    TranslationDataset,
    make_collate,
    make_pairs,
    greedy_decode,
    beam_search_decode,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # --- Model / training config ------------------------------------------
    # Sized to comfortably fit a 4GB GPU (GTX 1650) in fp32. Raise these if
    # you're back on a bigger card (e.g. D_MODEL=512, NUM_LAYERS=6 for a T4).
    MAX_LEN = 32
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    D_FF = 1024
    BATCH_SIZE = 32
    ACC_STEPS = 4       # effective batch size = BATCH_SIZE * ACC_STEPS = 128
    NUM_EPOCHS = 25
    WARMUP_STEPS = 3000
    NUM_MERGES = 37000

    VAL_SIZE = 0.05
    TEST_SIZE = 0.05
    SPLIT_SEED = 42

    # --- FIX 1: paths -------------------------------------------------------
    # Point these at your actual CSV files (each needs "en" and "de" columns)
    # and wherever you want the tokenizer saved/loaded from.
    DATA_DIR = "./data"
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    tokenizer_path = "tokenizer.json"

    CKPT_PATH = "last_checkpoint.pt"
    BEST_MODEL_PATH = "best_english_to_german_p2.pt"
    HISTORY_PATH = "training_history_pt2.csv"

    train_df = pd.read_csv("train.csv", lineterminator="\n")
    test_df = pd.read_csv("test.csv", lineterminator="\n")
    val_df = pd.read_csv("val.csv", lineterminator="\n")

    english_sentences = train_df["en"].tolist()
    german_sentences = train_df["de"].tolist()

    # --- Tokenizer setup with power-failure protection ---------------------
    temp_tokenizer_path = tokenizer_path + ".tmp"

    if os.path.exists(tokenizer_path):
        tok = Tokenizer.from_file(tokenizer_path)
        print("Loading from existing tokenizer")
    elif os.path.exists(temp_tokenizer_path):
        os.replace(temp_tokenizer_path, tokenizer_path)
        tok = Tokenizer.from_file(tokenizer_path)
        print("Recovered and loaded tokenizer from temp file")
    else:
        print("Training new tokenizer...")
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()

        corpus = english_sentences + german_sentences

        # vocab_size = special_tokens + base_alphabet + merges. The base
        # alphabet here is every unique character actually in the corpus
        # (not 256 -- that's a byte-level-BPE convention and we're using a
        # character-level Whitespace pre-tokenizer). Compute it directly so
        # NUM_MERGES actually maps to ~NUM_MERGES real merges instead of
        # guessing a buffer.
        alphabet_size = len(set("".join(corpus)))
        num_special_tokens = 4  # <pad>, <unk>, <s>, </s>
        total_vocab_size = NUM_MERGES + alphabet_size + num_special_tokens
        print(f"Corpus alphabet size: {alphabet_size} unique chars -> target vocab_size={total_vocab_size}")

        trainer = BpeTrainer(
            vocab_size=total_vocab_size,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
            show_progress=True,
        )

        try:
            tok.train_from_iterator(corpus, trainer=trainer)

            tok.save(temp_tokenizer_path)
            os.replace(temp_tokenizer_path, tokenizer_path)
            print("Tokenizer successfully trained and saved safely.")
        except Exception as e:
            print("Training interrupted. Cleaning up to prevent corruption...")
            if os.path.exists(temp_tokenizer_path):
                os.remove(temp_tokenizer_path)
            raise e

    vocab_size = tok.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    def filter_by_length(df, tok, max_len, src_col="en", tgt_col="de"):
        def within_limit(en, de):
            return len(tok.encode(en).ids) <= max_len - 2 and len(tok.encode(de).ids) <= max_len - 2

        mask = [
            within_limit(en, de)
            for en, de in tqdm(
                zip(df[src_col], df[tgt_col]), total=len(df), desc=f"Filtering (max_len={max_len})"
            )
        ]
        kept = df[mask].reset_index(drop=True)
        print(f"Kept {len(kept)}/{len(df)} pairs after filtering to max_len={max_len}")
        return kept

    train_df = filter_by_length(train_df, tok, MAX_LEN)
    val_df = filter_by_length(val_df, tok, MAX_LEN)
    test_df = filter_by_length(test_df, tok, MAX_LEN)

    pad_id = tok.token_to_id("<pad>")
    collate_fn = make_collate(pad_id)

    # --- Dataset caching -----------------------------------------------------
    def load_or_create_pairs(df, tok, cache_name):
        if os.path.exists(cache_name):
            print(f"Loading cached dataset from {cache_name}...")
            return torch.load(cache_name)
        else:
            print(f"Tokenizing {cache_name}...")
            examples = make_pairs(df, tok)
            torch.save(examples, cache_name)
            return examples

    train_examples = load_or_create_pairs(train_df, tok, "cache_train.pt")
    train_dataset = TranslationDataset(train_examples)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, collate_fn=collate_fn
    )

    val_examples = load_or_create_pairs(val_df, tok, "cache_val.pt")
    val_dataset = TranslationDataset(val_examples)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, collate_fn=collate_fn
    )

    test_examples = load_or_create_pairs(test_df, tok, "cache_test.pt")
    test_dataset = TranslationDataset(test_examples)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, collate_fn=collate_fn
    )

    encoder = Encoder(vocab_size, D_MODEL, NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, pad_id=pad_id).to(device)
    decoder = Decoder(vocab_size, D_MODEL, NUM_HEADS, num_layers=NUM_LAYERS, d_ff=D_FF, pad_id=pad_id).to(device)
    output_layer = make_output(D_MODEL, vocab_size).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(output_layer.parameters())
    total_params = sum(p.numel() for p in params)
    print(f"Total model parameters: {total_params:,}\n")

    optimizer = torch.optim.Adam(params, lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: scheduler_(step, D_MODEL, warmup_steps=WARMUP_STEPS)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    best_loss = float("inf")

    start_epoch = 0
    history = []

    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        output_layer.load_state_dict(ckpt["output_layer"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]

        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
        if os.path.exists(HISTORY_PATH):
            history = pd.read_csv(HISTORY_PATH).to_dict("records")

    print("Training...")
    for epoch in range(start_epoch, NUM_EPOCHS):
        start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{NUM_EPOCHS}", leave=True)
        optimizer.zero_grad()  # start of epoch, clean slate

        for step, batch in enumerate(pbar):
            loss, correct, total = train_one_batch(
                batch, encoder, decoder, output_layer, optimizer, criterion, device,
                accum_steps=ACC_STEPS,
            )
            if total > 0:
                total_loss += loss
                valid_batches += 1
            total_correct += correct
            total_tokens += total

            if (step + 1) % ACC_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix(
                loss=f"{total_loss / max(valid_batches, 1):.4f}",
                acc=f"{100 * total_correct / max(total_tokens, 1):.2f}%",
                lr=f"{current_lr:.6f}",
            )

        # flush any leftover accumulated gradients that didn't hit a full
        # ACC_STEPS boundary at the end of the epoch
        if len(train_loader) % ACC_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(valid_batches, 1)
        acc = total_correct / max(total_tokens, 1) * 100
        val_loss = evaluate(val_loader, encoder, decoder, output_layer, criterion, device)
        elapsed = time.time() - start

        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {acc:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "lr": current_lr,
            "time_seconds": elapsed,
        })

        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        torch.save({
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "output_layer": output_layer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "loss": best_loss,
            "config": {
                "vocab_size": vocab_size,
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "d_ff": D_FF,
            },
        }, CKPT_PATH)

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "output_layer": output_layer.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "config": {
                    "vocab_size": vocab_size,
                    "d_model": D_MODEL,
                    "num_heads": NUM_HEADS,
                    "num_layers": NUM_LAYERS,
                    "d_ff": D_FF,
                },
            }, BEST_MODEL_PATH)

            print(f"\u2713 Saved new best model (loss = {best_loss:.4f})")

    test_loss = evaluate(test_loader, encoder, decoder, output_layer, criterion, device)
    print(f"\nFinal test loss: {test_loss:.4f}")

    print("\n--- Sample translations (test set, never trained on) ---")
    for i in range(min(5, len(test_df))):
        en_sentence = test_df["en"].iloc[i]
        ge_reference = test_df["de"].iloc[i]

        src_ids = torch.tensor([tok.encode(en_sentence).ids])

        greedy_ids = greedy_decode(src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device)
        beam_ids = beam_search_decode(
            src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device, beam_width=4
        )

        if isinstance(greedy_ids, torch.Tensor):
            greedy_ids = greedy_ids.tolist()
        if isinstance(beam_ids, torch.Tensor):
            beam_ids = beam_ids.tolist()

        print(f"\nEN:    {en_sentence}")
        print(f"GE ref:{ge_reference}")
        print(f"Greedy:{tok.decode(greedy_ids, skip_special_tokens=True)}")
        print(f"Beam:  {tok.decode(beam_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
