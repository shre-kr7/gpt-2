# GPT-2 From Scratch (Character-Level)

A character-level GPT implementation built from scratch in PyTorch, following Andrej Karpathy's "Let's build GPT" curriculum. Trained on a custom dataset of three classic novels — *Alice in Wonderland*, *Pride and Prejudice*, and *The Adventures of Sherlock Holmes* (~1.5M characters combined) — rather than the standard Shakespeare dataset.

## Overview

This project implements a decoder-only Transformer language model, trained to predict the next character given a sequence of previous characters. It includes:

- Character-level tokenization (custom vocabulary built from the training text)
- Multi-head self-attention with causal masking
- Transformer blocks with pre-norm, residual connections, and a feed-forward network
- A training loop with train/validation loss tracking
- A text generation function that takes a prompt and produces new text

## Files

- `input.txt` — training corpus: the combined text of *Alice in Wonderland*, *Pride and Prejudice*, and *The Adventures of Sherlock Holmes*
- `new.py` — main script: data loading, model definition, training loop, and text generation
- `gpt.py` — Andrej Karpathy's original reference code, kept for comparison only (not the version actually run/trained)

## Requirements

```
torch
```

Install with:
```bash
pip install torch
```

(If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch from [pytorch.org](https://pytorch.org) instead of the default CPU-only build, so training runs on GPU.)

## How to Run

1. Place your training text in a file named `input.txt` in the same directory as the script.
2. Open `new.py` in VS Code (it uses `#%%` cell markers) or run it as a plain script:
   ```bash
   python new.py
   ```
3. The script will:
   - Build the character vocabulary from `input.txt`
   - Split the data into 90% train / 10% validation
   - Train the model for `max_iters` steps, printing train/val loss periodically
   - Generate a sample of text at the end

## Model Configuration

| Hyperparameter | Value |
|---|---|
| `block_size` (context length) | 32 |
| `n_embd` (embedding dimension) | 64 |
| `n_head` (attention heads) | 4 |
| `n_layer` (transformer blocks) | 4 |
| `batch_size` | 16 |
| `max_iters` | 5000 |
| `learning_rate` | 1e-3 |

Adjust these at the top of the script depending on your dataset size and available compute.

## Generating Text

After training, use the prompt-based generation helper to produce new text from a starting prompt:

```python
generate_text(prompt="Once upon a time", num_words=50)
```

This encodes the prompt, generates new characters, and trims the output down to the requested word count.

## Notes

- This is a character-level model, so vocabulary size depends on the unique characters in your dataset (not a fixed BPE vocabulary like GPT-2's actual 50k-token tokenizer).
- Expect validation loss in the 1.4–1.6 range at this scale/dataset size; getting much lower typically requires more data or a larger model.
- Training on CPU is significantly slower than GPU — if `torch.cuda.is_available()` returns `False` despite having a GPU, reinstall PyTorch with the correct CUDA build.

## Acknowledgments

Built following [Andrej Karpathy's "Let's build GPT" video](https://www.youtube.com/watch?v=kCc8FmEb1nY) and his nanoGPT repository.
