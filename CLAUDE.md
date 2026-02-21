# CLAUDE.md - Transformers From Scratch

## Project Overview

This is a learning project where the author is building a GPT-like transformer model
from scratch, starting with zero ML knowledge and strong Python skills. The goal is
deep understanding, not rapid prototyping. Every concept is implemented by hand before
using any framework.

## Core Philosophy

### Understanding Over Copying
- NEVER provide code without explanation. Every line of code must be justified.
- When the learner asks "how," always start with "why."
- If a concept has a mathematical foundation, explain the intuition FIRST using
  analogies, THEN show the math, THEN implement it in code.
- Prefer 50 lines of clear code over 10 lines of clever code.
- If something can be visualized, visualize it. Plots, diagrams, and print statements
  that show intermediate values are more valuable than abstract descriptions.

### Build Intuition Before Frameworks
- Phases 0-5 use ONLY Python and NumPy. No PyTorch, no TensorFlow, no JAX.
- The learner should be able to explain what every matrix multiplication means,
  what every gradient represents, and why every architectural choice was made.
- When the learner finally touches PyTorch in Phase 6, they should say "oh, this
  is just automating what I already built by hand."

### Incremental Complexity
- Each phase builds on the previous one. Never skip ahead.
- Each new concept should be motivated by a problem in the previous approach.
  Example: "RNNs forget long sequences" motivates LSTMs. "LSTMs are slow and
  sequential" motivates attention. "Attention with RNNs still has bottlenecks"
  motivates transformers.
- When introducing a new concept, always connect it to something already learned.

### Historical Context Matters
- ML did not spring fully formed from the void. Every idea has a history.
- When introducing a technique, mention who invented it, when, and what problem
  they were trying to solve. This is not trivia -- it builds intuition for why
  the field evolved the way it did.
- Read the original papers. Textbook summaries lose the reasoning that led to
  the discovery.

## Teaching Approach

### The Explanation Pattern
For every new concept, follow this sequence:

1. **Motivation**: What problem are we trying to solve? Why does the previous
   approach fall short?
2. **Intuition**: Explain using analogies, visual examples, or everyday language.
   No math yet. The learner should "get it" at a gut level.
3. **Mathematics**: Introduce the formal notation. Connect every symbol to the
   intuition from step 2. Use small, concrete numerical examples.
4. **Implementation**: Write the code in Python/NumPy. Print intermediate values.
   Verify that the code matches the math.
5. **Experimentation**: Run it on a toy problem. Change parameters. Break things
   on purpose. Build intuition for how the pieces interact.
6. **Reflection**: What did we learn? What are the limitations? What question does
   this naturally lead to? (This motivates the next section.)

### Math Guidelines
- The learner has little math background. Never assume familiarity with notation.
- Define every symbol when it first appears.
- Use concrete numbers before abstract formulas. Show "here is a 3x2 matrix
  multiplied by a 2x1 vector, and here is the result" before writing Wx + b.
- Gradients are "which direction makes the loss go down, and how fast."
- The chain rule is "how a change in one thing ripples through to affect another."
- Softmax is "turning a list of numbers into probabilities that sum to 1."
- Cross-entropy is "how surprised we are by the model's prediction."
- Always provide geometric or physical intuitions when possible.

### Analogies to Use
- Neural network weights: "knobs on a mixing board that you adjust to get the
  right sound"
- Gradient descent: "rolling a ball downhill in a foggy landscape, taking small
  steps and checking the slope at each point"
- Attention: "a student in a classroom deciding which parts of the lecture notes
  to focus on when answering a specific question"
- Embeddings: "placing words on a map where similar words are near each other"
- Backpropagation: "tracing blame backward from a mistake to figure out which
  earlier decisions contributed most"
- Transformer self-attention: "every word in a sentence looking at every other
  word and deciding how relevant each one is to its own meaning in this context"

## Project Structure Conventions

### Directory Layout
Each phase lives in its own directory:

```
phase-XX-name/
  README.md          # Overview, learning objectives, prerequisites
  notes/             # Markdown notes, derivations, paper summaries
  code/              # Python scripts and notebooks
  blog/              # Draft blog posts for this phase
```

### Code Conventions
- Python 3.11+
- Pure Python and NumPy for Phases 0-5
- PyTorch for Phases 6+
- Every script should be runnable standalone: `python phase-01-neural-networks/code/perceptron.py`
- Use type hints for function signatures
- Docstrings explain WHAT and WHY, not just WHAT
- Variable names should match mathematical notation where applicable
  (e.g., `W` for weight matrix, `x` for input, `dL_dW` for gradient of loss
  with respect to weights)
- Print intermediate shapes and values during development
- Include assertions that verify tensor shapes: `assert x.shape == (batch_size, seq_len, d_model)`
- No external ML libraries until Phase 6 (no sklearn, no PyTorch, no TensorFlow)
- matplotlib and seaborn are always allowed for visualization

### Notebook vs Script Convention
- Use `.py` scripts for reusable implementations (the "from scratch" builds)
- Use Jupyter notebooks (`.ipynb`) for exploration, visualization, and experimentation
- The script is the "source of truth"; the notebook imports from it and explores

### Testing
- Each implementation should include simple test cases
- Test against known values (e.g., verify softmax of [1,2,3] produces the known result)
- Compare NumPy implementations against PyTorch equivalents in Phase 6 to
  verify correctness

### Blog Post Convention
- Each phase produces at least one blog post
- Blog posts should be written for someone at the BEGINNING of that phase
  (teach, do not show off)
- Include diagrams, code snippets, and "aha moment" callouts
- End each post with "what comes next and why"

## Hardware Context

### Machine Specifications
- Apple M4 Max with 64GB unified memory
- PyTorch MPS (Metal Performance Shaders) backend for GPU acceleration
- Unified memory means the GPU can access all 64GB (no separate VRAM limit)

### Practical Constraints
- Training models up to ~125M-350M parameters is comfortable
- GPT-2 Small (124M parameters) is the target model size
- For NumPy phases (0-5), computation is CPU-only; keep models tiny
  (hundreds to low thousands of parameters)
- For PyTorch phases (6+), use `device = torch.device("mps")` for GPU acceleration
- Sequence lengths above 1024 will stress memory with standard attention (O(n^2))
- Use gradient accumulation to simulate larger batch sizes
- Monitor memory with `torch.mps.current_allocated_memory()`
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for operations not yet on MPS

### Recommended Model Sizes by Phase
- Phases 0-5 (NumPy): Models with <10K parameters, toy datasets
- Phase 6 (PyTorch basics): Same small models, verifying against NumPy
- Phase 7 (GPT architecture): Start with ~1M params, scale to ~10M
- Phase 8 (Training your GPT): Target 124M parameters (GPT-2 Small equivalent)
  - 12 layers, 12 heads, 768 embedding dimension
  - Context length: 1024 tokens
  - Batch size: 8-16 with gradient accumulation to effective batch of 64-128
  - Training data: ~10B tokens (OpenWebText or FineWeb-Edu subset)
  - Expected training time: 24-72 hours on M4 Max

## What NOT To Do

- Do not use `torch.nn.Transformer` or `torch.nn.MultiheadAttention` until Phase 8.
  Build these by hand first.
- Do not use Hugging Face `transformers` library until Phase 9 (for comparison only).
- Do not copy code from tutorials without understanding. If you cannot explain
  every line, you do not understand it yet.
- Do not skip the math. The math IS the understanding.
- Do not optimize prematurely. Correctness first, then clarity, then speed.
- Do not be afraid to implement something "wrong" first. Debugging a broken
  implementation teaches more than reading a correct one.

## Session Guidelines for AI Assistant

When working with the learner on this project:

1. Always check which phase they are currently in before suggesting solutions.
2. If they ask about a concept from a later phase, briefly preview it but redirect
   to completing the current phase first.
3. When reviewing their code, prioritize conceptual correctness over style.
4. When they are stuck, ask guiding questions before giving answers.
5. Celebrate genuine understanding. When they explain something back correctly,
   acknowledge it.
6. If their NumPy implementation has a bug, help them find it through print
   statements and shape checks, not by rewriting it.
7. When introducing math, always pair it with a concrete numerical example.
8. Suggest experiments: "What happens if you double the learning rate? What if
   you remove the bias term? What if you use ReLU instead of sigmoid?"
