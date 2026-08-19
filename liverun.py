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

# BEAM_WIDTH = 4
# MAX_LEN = 32


# def load_trained_model(checkpoint_path="best_en_ko_model.pt", tokenizer_path="tokenizer.json", device="cpu"):
#     """Reload a previously-trained model + tokenizer WITHOUT retraining --
#     use this in a separate script/session once training is done."""
#     tok = Tokenizer()
#     tok.load(tokenizer_path)

#     ckpt = torch.load(checkpoint_path, map_location=device)
#     cfg = ckpt["config"]

#     encoder = Encoder(cfg["vocab_size"], cfg["d_model"], cfg["num_heads"], num_layers=cfg["num_layers"]).to(device)
#     decoder = Decoder(cfg["vocab_size"], cfg["d_model"], cfg["num_heads"], num_layers=cfg["num_layers"]).to(device)
#     output_layer = make_output(cfg["d_model"], cfg["vocab_size"]).to(device)

#     encoder.load_state_dict(ckpt["encoder"])
#     decoder.load_state_dict(ckpt["decoder"])
#     output_layer.load_state_dict(ckpt["output_layer"])

#     return encoder, decoder, output_layer, tok


# def translate(sentence, encoder, decoder, output_layer, tok, device):
#     src_tokens = tok.encode(sentence)
#     src_ids = torch.tensor([tok.tokens_to_ids(src_tokens)])

#     greedy_ids = greedy_decode(src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device)
#     beam_ids = beam_search_decode(
#         src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device, beam_width=BEAM_WIDTH
#     )

#     return ids_to_sentence(greedy_ids, tok), ids_to_sentence(beam_ids, tok)


# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     print("Loading model and tokenizer...")
#     try:
#         encoder, decoder, output_layer, tok = load_trained_model(device=device)
#     except FileNotFoundError as e:
#         print(f"Could not find checkpoint files: {e}")
#         print("Make sure en_ko_model.pt and en_ko_tokenizer.json are in this folder.")
#         return

#     print(f"Loaded. Using device={device}")
#     print("Type an English sentence to translate (or 'quit' to exit).\n")

#     while True:
#         sentence = input("EN > ").strip()
#         if sentence.lower() in ("quit", "exit"):
#             break
#         if not sentence:
#             continue

#         greedy_translation, beam_translation = translate(sentence, encoder, decoder, output_layer, tok, device)

#         print(f"Greedy > {greedy_translation}")
#         print(f"Beam   > {beam_translation}\n")


# if __name__ == "__main__":
#     main()

import math
import time
import torch
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# Your local imports
from embedding import *
from core import *
from misc import *
from training import *

# Hugging Face Tokenizer import
from tokenizers import Tokenizer

BEAM_WIDTH = 4
MAX_LEN = 32


def load_trained_model(checkpoint_path="best_english_to_german.pt", tokenizer_path="tokenizer_en_to_ge.json", device="cpu"):
    """Reload a previously-trained model + tokenizer WITHOUT retraining --
    use this in a separate script/session once training is done."""
    
    # 1. FIXED: Load Hugging Face Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    encoder = Encoder(cfg["vocab_size"], cfg["d_model"], cfg["num_heads"], num_layers=cfg["num_layers"]).to(device)
    decoder = Decoder(cfg["vocab_size"], cfg["d_model"], cfg["num_heads"], num_layers=cfg["num_layers"]).to(device)
    output_layer = make_output(cfg["d_model"], cfg["vocab_size"]).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    output_layer.load_state_dict(ckpt["output_layer"])

    return encoder, decoder, output_layer, tok


def translate(sentence, encoder, decoder, output_layer, tok, device):
    # 2. FIXED: Hugging Face API to get integer IDs
    src_ids = torch.tensor([tok.encode(sentence).ids])

    greedy_ids = greedy_decode(src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device)
    beam_ids = beam_search_decode(
        src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device, beam_width=BEAM_WIDTH
    )
    
    # Ensure they are lists before passing to HF decode
    if isinstance(greedy_ids, torch.Tensor):
        greedy_ids = greedy_ids.tolist()
    if isinstance(beam_ids, torch.Tensor):
        beam_ids = beam_ids.tolist()

    # 3. FIXED: Use native Hugging Face decoding instead of ids_to_sentence
    greedy_translation = tok.decode(greedy_ids, skip_special_tokens=True)
    beam_translation = tok.decode(beam_ids, skip_special_tokens=True)

    return greedy_translation, beam_translation


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model and tokenizer...")
    try:
        encoder, decoder, output_layer, tok = load_trained_model(device=device)
    except FileNotFoundError as e:
        print(f"Could not find checkpoint files: {e}")
        print("Make sure best_english_to_german.pt and tokenizer_en_to_ge.json are in this folder.")
        return

    print(f"Loaded. Using device={device}")
    print("Type an English sentence to translate (or 'quit' to exit).\n")

    while True:
        sentence = input("EN > ").strip()
        if sentence.lower() in ("quit", "exit"):
            break
        if not sentence:
            continue

        greedy_translation, beam_translation = translate(sentence, encoder, decoder, output_layer, tok, device)

        print(f"Greedy > {greedy_translation}")
        print(f"Beam   > {beam_translation}\n")


if __name__ == "__main__":
    main()