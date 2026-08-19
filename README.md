```markdown
# English → German Transformer from Scratch

A from-scratch implementation of an **Encoder–Decoder Transformer** for English-to-German machine translation using PyTorch. 

This project implements the core Transformer architecture, custom BPE tokenization, dataset preparation, a custom training pipeline, greedy decoding, and beam-search decoding without relying on PyTorch's built-in `nn.Transformer` modules.

The model was explicitly designed and stabilized for **memory efficiency and numerical stability** on a low-VRAM GPU without Tensor Cores (e.g., a 4 GB NVIDIA GTX 1650).

---

## ✨ Features

- **Custom Encoder–Decoder**: Multi-Head Self-Attention, Cross-Attention, and Feed-Forward networks written from scratch.
- **4GB GPU Optimized**: Model dimensions scaled down (`d_model=256`, 4 layers) and uses Gradient Accumulation (effective batch size of 128) to prevent OOM errors.
- **Extreme Training Stability**: 
  - **Pre-Norm Architecture:** LayerNorm applied *before* attention/FFN to prevent gradient explosion.
  - **NaN Guard:** Custom detection for fully-masked attention rows to prevent `softmax(-inf)` NaN poisoning.
  - **Pure FP32 Training:** Intentional exclusion of mixed-precision (AMP) to avoid FP16 overflow on non-Tensor Core GPUs.
- **Custom Tokenizer**: Trains a custom BPE tokenizer with a character-level whitespace pre-tokenizer directly on the training corpus.
- **Decoding Algorithms**: Supports both Greedy Decoding and Length-Normalized Beam Search.
- **Visualization**: Automated training history tracking and plotting.

---

## 🏗️ Architecture

```text
                    English Sentence
                           │
                           ▼
                   Custom BPE Tokenizer
                           │
                           ▼
                    Token Embeddings
                           │
                           ▼
                  Positional Encoding
                           │
                           ▼
              ┌────────────────────────┐
              │        ENCODER         │
              │                        │
              │   Multi-Head Attention │
              │            ↓           │
              │     Feed Forward       │
              │            ↓           │
              │          × 4           │
              └───────────┬────────────┘
                          │
                    Encoder Output
                          │
                          ▼
              ┌────────────────────────┐
              │        DECODER         │
              │                        │
              │   Masked Self-Attn     │
              │            ↓           │
              │    Cross Attention     │
              │            ↓           │
              │     Feed Forward       │
              │            ↓           │
              │          × 4           │
              └───────────┬────────────┘
                          │
                          ▼
                    Linear Output
                          │
                          ▼
                   German Tokens

```

---

## 🧠 Numerical Stability & GPU Constraints

Because this Transformer is implemented from scratch and targeted at older/smaller GPUs (like the GTX 1650), textbook implementations often fail with NaN losses. The following protections were built into `core.py` and `main.py`:

### 1. Pre-Normalization

The model uses Pre-Norm instead of Post-Norm:

* `x → LayerNorm → Attention → Residual`
* `x → LayerNorm → Feed Forward → Residual`
This keeps gradients much better behaved early in training and is the single biggest lever against NaN loss in from-scratch Transformers.

### 2. Fully-Masked Attention Protection

When applying padding masks, a query position might have no valid keys to attend to. The raw scores row becomes entirely `-inf`, and `softmax(-inf, -inf, ...)` equals `NaN`. The code detects fully-masked rows and zeros them out *before* softmax, replacing them with a harmless uniform distribution that gets discarded downstream.

### 3. Pure FP32 Training

The GTX 1650 (TU117 architecture) lacks Tensor Cores. PyTorch's `autocast()` (FP16 mixed precision) provides zero speedup on this hardware and introduces severe risks of FP16 overflow since attention logits can easily exceed FP16's ~65,504 limit. Training is locked to FP32.

### 4. Gradient Accumulation

To simulate large batch sizes on 4GB VRAM, the training loop uses `BATCH_SIZE = 32` with `ACC_STEPS = 4`, achieving an effective batch size of 128.

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `D_MODEL` | 256 | Embedding dimension |
| `NUM_HEADS` | 8 | Attention heads |
| `NUM_LAYERS` | 4 | Encoder and Decoder layers |
| `D_FF` | 1024 | Feed-forward hidden dimension |
| `MAX_LEN` | 32 | Maximum sequence length |
| `EFFECTIVE_BATCH` | 128 | 32 (batch) × 4 (accum steps) |
| `NUM_EPOCHS` | 25 | Total training epochs |
| `NUM_MERGES` | 37,000 | BPE Tokenizer merges |

---

## 📊 Training Results

The model was trained for **25 epochs**.

* **Final Training Loss**: ~3.95
* **Final Validation Loss**: ~3.93
* **Final Token Accuracy**: ~49.98%

*The plot above is automatically generated by `plotgraph.py` reading from the CSV history logs.*

---

## 🔤 Data Processing & Tokenization

The pipeline utilizes the HuggingFace `tokenizers` library to train a **Byte-Pair Encoding (BPE)** model on the combined English and German corpus.

Sequences are processed as follows to enable teacher forcing:

* **Encoder Input:** `[source tokens] + </s>`
* **Decoder Input:** `<s> + [target tokens]`
* **Training Label:** `[target tokens] + </s>`

Two attention masks are generated on the fly:

1. **Padding Mask:** Prevents attention to `<pad>` tokens.
2. **Causal (Autoregressive) Mask:** A lower-triangular matrix that prevents the decoder from "looking ahead" at future tokens.

---

## 📁 Project Structure

```text
.
├── core.py                       # Core Transformer architecture & utilities
├── main.py                       # Training loop, dataset collation, tokenization
├── demo.py                       # Interactive CLI translation interface
├── plotgraph.py                  # Generates training_curves.png from CSV logs
├── tokenizer.json                # Saved custom BPE tokenizer
├── best_english_to_german_p2.pt  # Saved model weights & config
├── final_history_training.csv    # Training metrics log
├── training_curves.png           # Accuracy/Loss visualization
└── README.md

```

---

## 🚀 Installation & Usage

### 1. Install Dependencies

```bash
pip install torch tokenizers pandas matplotlib tqdm

```

### 2. Training the Model

Ensure your data (`train.csv`, `val.csv`, `test.csv` containing `en` and `de` columns) is located in the `./data` directory, then run:

```bash
python3 main.py

```

*This will automatically train the tokenizer, filter sequences by length, begin the training loop, and save the best checkpoint.*

### 3. Plotting Training Curves

To generate or update the loss and accuracy graphs from the CSV logs:

```bash
python3 plotgraph.py

```

### 4. Interactive Translation (Inference)

Run the demo script to test the model dynamically. It will load the checkpoint and run both Greedy and Beam Search decoding:

```bash
python3 demo.py

```

**Example output:**

```text
Using device: cuda
Loaded checkpoint from 'best_english_to_german_p2.pt'

Type an English sentence to translate. Type 'exit' or 'quit' to stop.

EN> how are you today?
Greedy: wie geht es dir heute?
Beam:   wie geht es dir heute?

```

---

## 📜 License

This project is intended for educational and research purposes—specifically demonstrating how to overcome the hardware and numerical challenges of training large architectures from scratch.

```

```
