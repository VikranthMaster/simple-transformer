English → German Transformer from Scratch

A from-scratch implementation of an Encoder–Decoder Transformer for English-to-German machine translation using PyTorch. The project implements the Transformer architecture, tokenization, dataset preparation, training pipeline, greedy decoding, and beam-search decoding without relying on PyTorch's built-in Transformer modules.

The model was designed and trained with memory efficiency and numerical stability in mind, making it suitable for training on a 4 GB GPU such as the NVIDIA GTX 1650.

✨ Features

- Transformer Encoder–Decoder architecture implemented from scratch
- Multi-Head Self-Attention
- Cross-Attention between encoder and decoder
- Sinusoidal positional encoding
- Pre-Norm Transformer blocks for improved training stability
- Teacher-forcing during training
- Padding and causal attention masks
- Greedy decoding
- Beam-search decoding
- Custom dataset and batch collation pipeline
- No mixed-precision dependency
- NaN/Inf loss protection
- No PyTorch "nn.Transformer" used

🏗️ Architecture

The model follows the original Encoder–Decoder Transformer design:

English Sentence
│
▼
Tokenizer
│
▼
Source Embeddings
│
▼
Positional Encoding
│
▼
┌─────────────────────┐
│ Transformer │
│ Encoder │
│ │
│ Multi-Head Attention│
│ ↓ │
│ Feed Forward │
│ × N │
└─────────────────────┘
│
│ Encoder Output
▼
┌─────────────────────┐
│ Transformer │
│ Decoder │
│ │
│ Masked Self-Attn │
│ ↓ │
│ Cross-Attention │
│ ↓ │
│ Feed Forward │
│ × N │
└─────────────────────┘
│
▼
Linear Output Layer
│
▼
German Tokens

Core Components

- Encoder: Processes the complete source sentence and produces contextual representations.
- Decoder: Generates the German translation autoregressively.
- Multi-Head Attention: Allows the model to attend to different parts of the sequence simultaneously.
- Cross-Attention: Allows the decoder to attend to the encoder's representations.
- Feed-Forward Network: Applies a position-wise nonlinear transformation.
- Positional Encoding: Adds information about token positions.
- Layer Normalization: Uses pre-normalization to improve gradient stability.

🧠 Numerical Stability

Because this Transformer is implemented from scratch, additional care was taken to prevent unstable training.

Pre-Normalization

Each Transformer sub-layer applies LayerNorm before attention or feed-forward computation:

x → LayerNorm → Attention → Residual
x → LayerNorm → Feed Forward → Residual

This generally provides more stable gradient flow than post-normalization for deep or from-scratch Transformer implementations.

Fully Masked Attention Protection

Attention can produce NaNs when an entire attention row is masked:

softmax(-∞, -∞, -∞, ...) → NaN

The implementation detects fully masked rows and replaces the scores with zeros before applying softmax.

FP32 Training

The training pipeline intentionally uses standard FP32 instead of automatic FP16 mixed precision. This avoids unnecessary numerical instability on GPUs without Tensor Cores.

📊 Training Results

The following plot shows the training progress of the model:

"Training Curves" (training_curves.png)

The curves can be used to monitor:

- Training loss
- Validation loss
- Training/validation behavior over epochs
- Potential overfitting or instability

🔤 Data Processing

Each English/German sentence pair is converted into three sequences:

Encoder Input

[source tokens] + </s>

Decoder Input

<s> + [target tokens]

Training Label

[target tokens] + </s>

This enables standard teacher forcing, where the decoder receives the previous target tokens while learning to predict the next token.

Sequences are dynamically padded within each batch.

🎯 Attention Masks

The implementation uses two major types of masks.

Source Padding Mask

Prevents the encoder and decoder cross-attention from attending to padding tokens.

Causal Mask

Prevents the decoder from seeing future tokens during training.

For example:

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1

This forces the decoder to generate tokens autoregressively.

🔎 Decoding

Two decoding strategies are implemented.

Greedy Decoding

At every step, the token with the highest probability is selected:

next_token = argmax(P(token | previous_tokens))

This is fast but can produce suboptimal sequences.

Beam Search

Beam search maintains multiple candidate translations simultaneously.

The implementation supports configurable beam width:

BEAM_WIDTH = 4

A length-normalized score is used to reduce the tendency of beam search to prefer very short sequences.

📁 Project Structure

.
├── core.py
├── demo.py
├── tokenizer.json
├── best_english_to_german_p2.pt
├── training_curves.png
└── README.md

"core.py"

Contains the main implementation:

- Transformer encoder
- Transformer decoder
- Multi-head attention
- Feed-forward networks
- Positional encoding
- Dataset helpers
- Collation
- Training/evaluation functions
- Greedy decoding
- Beam search

"demo.py"

Interactive translation script for testing the trained model.

"tokenizer.json"

Tokenizer used for converting English and German text into token IDs.

"best_english_to_german_p2.pt"

Trained model checkpoint containing the encoder, decoder, output layer, and model configuration.

⚙️ Installation

Clone the repository and install the required dependencies:

git clone <your-repository-url>
cd <repository-name>

pip install torch
pip install tokenizers
pip install tqdm

🚀 Running the Translator

After placing the trained checkpoint and tokenizer in the project directory:

python3 demo.py

Then enter an English sentence:

EN> how are you today?

The model produces translations using both decoding strategies:

Greedy: wie geht es dir heute?
Beam: wie geht es dir heute?

Type "exit" or "quit" to stop the program.

💻 Hardware

The implementation was optimized with a low-VRAM GPU in mind.

Component| Configuration
Framework| PyTorch
GPU Target| NVIDIA GTX 1650 / similar 4 GB GPU
Precision| FP32
Architecture| Encoder–Decoder Transformer
Decoding| Greedy + Beam Search

🛠️ Design Decisions

This project intentionally avoids using a high-level Transformer implementation so that the individual components can be studied and controlled directly.

The implementation includes:

- Custom attention calculations
- Custom encoder/decoder layers
- Explicit masking
- Explicit residual connections
- Custom training loop
- Custom decoding algorithms

This makes the project useful for understanding how Transformers work internally, rather than treating the Transformer as a black-box module.

📌 Future Improvements

- [ ] Add BLEU / SacreBLEU evaluation
- [ ] Add label smoothing
- [ ] Add learning-rate visualization
- [ ] Add checkpoint resume support
- [ ] Improve beam-search length normalization
- [ ] Add batch inference
- [ ] Experiment with larger datasets
- [ ] Experiment with SentencePiece/BPE tokenizer configurations
- [ ] Compare against PyTorch's built-in Transformer implementation

📜 License

This project is intended for educational and research purposes.
