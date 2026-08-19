# import math
# import time
# import torch
# import pandas as pd
# import torch.nn as nn
# from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import Dataset, DataLoader
# from tokenizer import *
# from embedding import *
# from core import *
# from misc import *
# from training import *
# from tqdm.auto import tqdm
# import os
# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# from tokenizers.pre_tokenizers import Whitespace



# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using Device: {device}")
#     MAX_LEN = 32
#     D_MODEL = 512
#     NUM_HEADS = 8
#     NUM_LAYERS = 6
#     BATCH_SIZE = 16
#     NUM_EPOCHS = 20
#     WARMUP_STEPS = 4000
#     NUM_MERGES = 37000


#     VAL_SIZE = 0.05     
#     TEST_SIZE = 0.05
#     SPLIT_SEED = 42

#     train_df = pd.read_csv("train.csv", lineterminator="\n")
#     test_df = pd.read_csv("test.csv", lineterminator="\n")
#     val_df = pd.read_csv("val.csv", lineterminator="\n")


#     english_sentences = train_df["en"].tolist()
#     german_sentences = train_df["de"].tolist()
#     # tok = Tokenizer()
#     # if os.path.exists("tokenizer_en_ge.json"):
#     #     tok.load("tokenizer_en_ge.json")
#     #     print("Loaded existing tokenizer")
#     # else:
#     #     corpus = english_sentences + german_sentences
#     #     tok.bpe(corpus, num_merges=NUM_MERGES, printing=True, save_every=10)
#     #     tok.save("tokenizer_en_ge.json")
#     # print(f"Vocab size: {len(tok.token_to_id)}")

#     tokenizer_path = "tokenizer_en_to_ge.json"
#     if os.path.exists(tokenizer_path):
#         tok = Tokenizer(BPE(unk_token="<unk>"))
#         print("Loading from existing tokenizer")
#     else:
#         tok = Tokenizer(BPE(unk_token="<unk>"))
#         tok.pre_tokenizer = Whitespace()
#         total_vocab_size = NUM_MERGES + 256
#         trainer = BpeTrainer(
#             vocab_size = total_vocab_size,
#             special_tokens=["<pad>", "<unk>", "<s>", "</s>"], # Add any other special tokens you need
#             show_progress=True
#         )
#         corpus = english_sentences + german_sentences
#         tok.train_from_iterator(corpus, trainer=trainer)
#         tok.save(tokenizer_path)

#     print(f"Vocab size: {tok.get_vocab_size()}")

#     pad_id = tok.token_to_id("<pad>")
#     collate_fn = make_collate(pad_id)

#     train_examples = make_pairs(train_df, tok)
#     train_dataset = TranslationDataset(train_examples)
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

#     val_examples = make_pairs(val_df, tok)
#     val_dataset = TranslationDataset(val_examples)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=collate_fn)

#     test_examples = make_pairs(test_df, tok)
#     test_dataset = TranslationDataset(test_examples)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,collate_fn=collate_fn)


#     vocab_size = len(tok.token_to_id)
#     encoder = Encoder(vocab_size, D_MODEL, NUM_HEADS, num_layers=NUM_LAYERS).to(device)
#     decoder = Decoder(vocab_size, D_MODEL, NUM_HEADS, num_layers=NUM_LAYERS).to(device)
#     output_layer = make_output(D_MODEL, vocab_size).to(device)


#     params = list(encoder.parameters()) + list(decoder.parameters()) + list(output_layer.parameters())
#     total_params = sum(p.numel() for p in params)
#     print(f"Total model parameters: {total_params:,}\n")


#     optimizer = torch.optim.Adam(params, lr=1.0, betas=(0.9, 0.98), eps=1e-9)
#     scheduler = torch.optim.lr_scheduler.LambdaLR(
#         optimizer, lr_lambda=lambda step: scheduler_(step, D_MODEL, warmup_steps=WARMUP_STEPS)
#     )
#     criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
#     best_loss = float("inf")

#     start_epoch = 0
#     CKPT_PATH = "last_checkpoint.pt"

#     if os.path.exists(CKPT_PATH):
#         ckpt = torch.load(CKPT_PATH, map_location=device)
#         encoder.load_state_dict(ckpt["encoder"])
#         decoder.load_state_dict(ckpt["decoder"])
#         output_layer.load_state_dict(ckpt["output_layer"])
#         optimizer.load_state_dict(ckpt["optimizer"])
#         scheduler.load_state_dict(ckpt["scheduler"])
#         start_epoch = ckpt["epoch"] + 1
#         best_loss = ckpt["loss"]
#         print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")


#     print("Training...")
#     for epoch in range(start_epoch, NUM_EPOCHS):
#         start = time.time()
#         total_loss = 0.0
#         total_correct = 0
#         total_tokens = 0

#         pbar = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{NUM_EPOCHS}", leave=True)

#         for batch in pbar:
#             loss,correct, total = train_one_batch(batch, encoder, decoder, output_layer, optimizer, criterion, device)
#             scheduler.step()
#             total_loss += loss
#             total_correct += correct
#             total_tokens += total

#             pbar.set_postfix(
#                 loss=f"{total_loss / (pbar.n + 1):.4f}",
#                 acc=f"{100 * total_correct / total_tokens:.2f}%",
#                 lr=f"{scheduler.get_last_lr()[0]:.6f}"
#             )

#         avg_loss = total_loss / len(train_loader)
#         acc = total_correct / total_tokens * 100
#         val_loss = evaluate(val_loader, encoder, decoder, output_layer, criterion, device)
#         elapsed = time.time() - start
#         print(
#             f"Epoch {epoch+1}/{NUM_EPOCHS} | "
#             f"Train Loss: {avg_loss:.4f} | "
#             f"Val Loss: {val_loss:.4f} |"
#             f"Accuracy: {acc:.2f}% | "
#             f"LR: {scheduler.get_last_lr()[0]:.6f} | "
#             f"Time: {elapsed:.1f}s"
#         )
#         torch.save({
#             "encoder": encoder.state_dict(),
#             "decoder": decoder.state_dict(),
#             "output_layer": output_layer.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "scheduler": scheduler.state_dict(),
#             "epoch": epoch,
#             "loss": best_loss,
#             "config": {
#                 "vocab_size": vocab_size,
#                 "d_model": D_MODEL,
#                 "num_heads": NUM_HEADS,
#                 "num_layers": NUM_LAYERS,
#             },
#         }, CKPT_PATH)

#         if val_loss < best_loss:
#             best_loss = val_loss
            
#             torch.save({
#                 "encoder": encoder.state_dict(),
#                 "decoder": decoder.state_dict(),
#                 "output_layer": output_layer.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "scheduler": scheduler.state_dict(),
#                 "epoch": epoch,
#                 "loss": best_loss,
#                 "config": {
#                     "vocab_size": vocab_size,
#                     "d_model": D_MODEL,
#                     "num_heads": NUM_HEADS,
#                     "num_layers": NUM_LAYERS,
#                 },
#             }, "best_english_to_german.pt")

#             print(f"✓ Saved new best model (loss = {best_loss:.4f})")

#     test_loss = evaluate(test_loader, encoder, decoder, output_layer, criterion, device)
#     print(f"\nFinal test loss: {test_loss:.4f}")

#     # --- Inference: sample translations from the held-out TEST set ---
#     print("\n--- Sample translations (test set, never trained on) ---")
#     for i in range(min(5, len(test_df))):
#         en_sentence = test_df["en"].iloc[i]
#         ko_reference = test_df["de"].iloc[i]

#         src_tokens = tok.encode(en_sentence)
#         src_ids = torch.tensor([tok.tokens_to_ids(src_tokens)])

#         greedy_ids = greedy_decode(src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device)
#         beam_ids = beam_search_decode(src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device, beam_width=4)

#         print(f"\nEN:    {en_sentence}")
#         print(f"Ge ref:{ko_reference}")
#         print(f"Greedy:{ids_to_sentence(greedy_ids, tok)}")
#         print(f"Beam:  {ids_to_sentence(beam_ids, tok)}")

# if __name__ == "__main__":
#     main()


import os
import math
import time
import torch
import shutil
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from tokenizer import *
from embedding import *
from core import *
from misc import *
from training import *

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")
    
    MAX_LEN = 32
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    BATCH_SIZE = 16
    ACC_STEPS = 8
    NUM_EPOCHS = 10
    WARMUP_STEPS = 3000
    NUM_MERGES = 37000

    VAL_SIZE = 0.05     
    TEST_SIZE = 0.05
    SPLIT_SEED = 42
    tokenizer_path = "tokenizer_en_to_ge.json"

    CKPT_PATH = "last_checkpoint.pt"
    BEST_MODEL_PATH = "best_english_to_german.pt"
    HISTORY_PATH = "training_history.csv"

    train_df = pd.read_csv("train.csv", lineterminator="\n")
    test_df = pd.read_csv("test.csv", lineterminator="\n")
    val_df = pd.read_csv("val.csv", lineterminator="\n")

    english_sentences = train_df["en"].tolist()
    german_sentences = train_df["de"].tolist()

    # --- Tokenizer Setup with Power Failure Protection ---
    temp_tokenizer_path = tokenizer_path + ".tmp"

    if os.path.exists(tokenizer_path):
        tok = Tokenizer.from_file(tokenizer_path)
        print("Loading from existing tokenizer")
    elif os.path.exists(temp_tokenizer_path):
        # Recover from a power failure that happened right after saving the temp file
        os.replace(temp_tokenizer_path, tokenizer_path)
        tok = Tokenizer.from_file(tokenizer_path)
        print("Recovered and loaded tokenizer from temp file")
    else:
        print("Training new tokenizer...")
        tok = Tokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        total_vocab_size = NUM_MERGES + 256
        
        trainer = BpeTrainer(
            vocab_size = total_vocab_size,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
            show_progress=True
        )
        
        try:
            corpus = english_sentences + german_sentences
            tok.train_from_iterator(corpus, trainer=trainer)
            
            # Atomic save: save to temp first, then rename
            tok.save(temp_tokenizer_path)
            os.replace(temp_tokenizer_path, tokenizer_path)
            print("Tokenizer successfully trained and saved safely.")
        except Exception as e:
            print(f"Training interrupted. Cleaning up to prevent corruption...")
            if os.path.exists(temp_tokenizer_path):
                os.remove(temp_tokenizer_path)
            raise e

    vocab_size = tok.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    pad_id = tok.token_to_id("<pad>")
    collate_fn = make_collate(pad_id)

    # --- DATASET CACHING LOGIC ---
    def load_or_create_pairs(df, tok, cache_name):
        if os.path.exists(cache_name):
            print(f"Loading cached dataset from {cache_name}...")
            return torch.load(cache_name)
        else:
            print(f"Tokenizing {cache_name}...")
            examples = make_pairs(df, tok)
            torch.save(examples, cache_name)
            return examples

    train_examples = load_or_create_pairs(train_df, tok, "cache_train_v1.pt")
    train_dataset = TranslationDataset(train_examples)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, collate_fn=collate_fn)

    val_examples = load_or_create_pairs(val_df, tok, "cache_val_v2.pt")
    val_dataset = TranslationDataset(val_examples)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    test_examples = load_or_create_pairs(test_df, tok, "cache_test_v3.pt")
    test_dataset = TranslationDataset(test_examples)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(vocab_size, D_MODEL, NUM_HEADS, num_layers=NUM_LAYERS).to(device)
    decoder = Decoder(vocab_size, D_MODEL, NUM_HEADS, num_layers=NUM_LAYERS).to(device)
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
    history = [] # FIX: Initialize history list BEFORE checking checkpoint

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
            history = pd.read_csv(HISTORY_PATH).to_dict('records')

        print("Training...")
        for epoch in range(start_epoch, NUM_EPOCHS):
            start = time.time()
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0

            pbar = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{NUM_EPOCHS}", leave=True)
            optimizer.zero_grad()  # start of epoch, clean slate

            for step, batch in enumerate(pbar):
                loss, correct, total = train_one_batch(
                    batch, encoder, decoder, output_layer, optimizer, criterion, device,
                    accum_steps=ACC_STEPS
                )
                total_loss += loss
                total_correct += correct
                total_tokens += total

                if (step + 1) % ACC_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                current_lr = scheduler.get_last_lr()[0]

                pbar.set_postfix(
                    loss=f"{total_loss / (pbar.n + 1):.4f}",
                    acc=f"{100 * total_correct / total_tokens:.2f}%",
                    lr=f"{current_lr:.6f}"
                )

            # flush any leftover accumulated gradients if len(train_loader) isn't
            # a clean multiple of ACC_STEPS
            if (step + 1) % ACC_STEPS != 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
        avg_loss = total_loss / len(train_loader)
        acc = total_correct / total_tokens * 100
        val_loss = evaluate(val_loader, encoder, decoder, output_layer, criterion, device)
        elapsed = time.time() - start
        
        # Ensure we have current_lr for the end of the epoch print/history
        current_lr = scheduler.get_last_lr()[0] 
        
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
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
            "lr": current_lr, # FIX: Now this variable actually exists!
            "time_seconds": elapsed
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
                },
            }, BEST_MODEL_PATH)

            print(f"✓ Saved new best model (loss = {best_loss:.4f})")

    test_loss = evaluate(test_loader, encoder, decoder, output_layer, criterion, device)
    print(f"\nFinal test loss: {test_loss:.4f}")

    # --- Inference: sample translations from the held-out TEST set ---
    print("\n--- Sample translations (test set, never trained on) ---")
    for i in range(min(5, len(test_df))):
        en_sentence = test_df["en"].iloc[i]
        ge_reference = test_df["de"].iloc[i]

        # Hugging Face Tokenizer API change for encoding
        src_ids = torch.tensor([tok.encode(en_sentence).ids])

        greedy_ids = greedy_decode(src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device)
        beam_ids = beam_search_decode(src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device, beam_width=4)

        # Ensure greedy_ids and beam_ids are lists before decoding
        if isinstance(greedy_ids, torch.Tensor):
            greedy_ids = greedy_ids.tolist()
        if isinstance(beam_ids, torch.Tensor):
            beam_ids = beam_ids.tolist()

        print(f"\nEN:    {en_sentence}")
        print(f"GE ref:{ge_reference}")
        # Hugging Face Tokenizer API change for decoding (replaces ids_to_sentence)
        print(f"Greedy:{tok.decode(greedy_ids, skip_special_tokens=True)}")
        print(f"Beam:  {tok.decode(beam_ids, skip_special_tokens=True)}")

if __name__ == "__main__":
    main()