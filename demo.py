"""
translate.py

Interactive EN -> DE translator using the trained checkpoint.

Usage:
    python3 translate.py
    EN> how are you today?
    Greedy: wie geht es dir heute?
    Beam:   wie geht es dir heute?
    EN> exit
"""

import torch

from tokenizers import Tokenizer
from core import Encoder, Decoder, make_output, greedy_decode, beam_search_decode

MODEL_PATH = "best_english_to_german_p2.pt"   # or "last_checkpoint.pt" to use the latest instead of best
TOKENIZER_PATH = "tokenizer.json"
MAX_LEN = 32          # keep in sync with training
BEAM_WIDTH = 4


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    cfg = ckpt["config"]

    encoder = Encoder(
        cfg["vocab_size"], cfg["d_model"], cfg["num_heads"],
        num_layers=cfg["num_layers"], d_ff=cfg.get("d_ff"),
    ).to(device)
    decoder = Decoder(
        cfg["vocab_size"], cfg["d_model"], cfg["num_heads"],
        num_layers=cfg["num_layers"], d_ff=cfg.get("d_ff"),
    ).to(device)
    output_layer = make_output(cfg["d_model"], cfg["vocab_size"]).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    output_layer.load_state_dict(ckpt["output_layer"])

    encoder.eval()
    decoder.eval()
    output_layer.eval()

    print(f"Loaded checkpoint from '{MODEL_PATH}' (epoch {ckpt.get('epoch', '?')}, "
          f"val loss {ckpt.get('loss', float('nan')):.4f})")

    return encoder, decoder, output_layer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tok = Tokenizer.from_file(TOKENIZER_PATH)
    encoder, decoder, output_layer = load_model(device)

    print("\nType an English sentence to translate. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            text = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            print("Bye.")
            break

        src_ids = torch.tensor([tok.encode(text).ids])

        greedy_ids = greedy_decode(
            src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device
        )
        beam_ids = beam_search_decode(
            src_ids, encoder, decoder, output_layer, tok, max_len=MAX_LEN, device=device,
            beam_width=BEAM_WIDTH,
        )

        if isinstance(greedy_ids, torch.Tensor):
            greedy_ids = greedy_ids.tolist()
        if isinstance(beam_ids, torch.Tensor):
            beam_ids = beam_ids.tolist()

        print(f"Greedy: {tok.decode(greedy_ids, skip_special_tokens=True)}")
        print(f"Beam:   {tok.decode(beam_ids, skip_special_tokens=True)}\n")


if __name__ == "__main__":
    main()
