# Transformers From Scratch

Building a GPT-like language model from absolute scratch -- starting with pure Python and NumPy, graduating to PyTorch only after every component is understood by hand.

## What This Is

A learning project that traces the full history of neural language models:

**Perceptron (1958) → MLP → RNN → LSTM → Attention → Transformer (2017) → GPT → Modern LLMs**

Each concept is implemented from scratch, motivated by the limitations of what came before. The final goal is training a 124M parameter GPT-2 Small equivalent on an Apple M4 Max (64GB).

## Project Structure

```
phase-00-math-foundations/       # Linear algebra, calculus, probability
phase-01-neural-networks/        # Perceptron, MLP, backpropagation
phase-02-sequence-models/        # RNN, LSTM, GRU, Seq2Seq
phase-03-text-representation/    # Tokenizers, embeddings, Word2Vec, BPE
phase-04-attention/              # Bahdanau, Luong, self-attention, multi-head
phase-05-transformer/            # Full transformer in NumPy
phase-06-pytorch-transition/     # Rebuild in PyTorch, verify against NumPy
phase-07-gpt-architecture/       # Decoder-only GPT in PyTorch
phase-08-training-gpt/           # Train 124M param model on M4 Max
phase-09-modern-developments/    # RoPE, Flash Attention, RLHF, LLaMA, Mistral
phase-10-blog-writing/           # Blog posts documenting the journey
papers/                          # Reading list and paper notes
```

Each phase contains `notes/`, `code/`, and `blog/` subdirectories.

## Principles

- **Understanding over copying**: Every line of code must be justified
- **Build intuition before frameworks**: Phases 0-5 use only Python + NumPy
- **Historical context matters**: Know who built what, when, and why
- **The math is the understanding**: Concrete numbers before abstract formulas

See [CLAUDE.md](CLAUDE.md) for full project guidelines and [ROADMAP.md](ROADMAP.md) for the detailed learning plan.

## Hardware

- Apple M4 Max, 64GB unified memory
- PyTorch MPS backend (Phase 6+)
- Target: GPT-2 Small (124M params, 1024 context, 50K vocab)

## Papers

35 papers from Rosenblatt (1958) to Mistral (2023), organized by topic in the roadmap. The core paper is [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).
