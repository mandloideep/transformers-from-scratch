# Transformers From Scratch: Learning Roadmap

> From zero ML knowledge to building your own GPT, one concept at a time.

## How to Use This Roadmap

Each phase has:
- **Key Concepts**: What you will learn
- **Papers to Read**: Original research, with full citations
- **Actionable Items**: Code to write, experiments to run
- **Blog Post Topic**: What to write about after completing the phase
- **Prerequisites**: What must be finished before starting

Work through the phases in order. Do not skip ahead. Each phase solves a problem
that motivates the next one. If you skip Phase 2 (sequence models), you will not
understand why attention was invented. If you skip Phase 4 (attention), the
transformer paper will feel arbitrary instead of brilliant.

---

## Phase 0: Mathematical Foundations

**Goal**: Build the mathematical vocabulary needed for everything that follows.
You do not need to become a mathematician. You need to understand what the
operations mean and be able to implement them.

**Prerequisites**: Python proficiency. Nothing else.

### Key Concepts

#### Linear Algebra
- **Scalars, vectors, and matrices**: A scalar is a single number. A vector is a
  list of numbers (think of it as a point or direction in space). A matrix is a
  grid of numbers (think of it as a transformation that moves points around).
- **Dot product**: Measures how similar two vectors are. If two vectors point in
  the same direction, their dot product is large and positive. If perpendicular,
  it is zero. This is the fundamental operation behind attention.
- **Matrix multiplication**: Applying a transformation. When we multiply a weight
  matrix W by an input vector x, we are transforming x into a new representation.
  This is what neural networks do at every layer.
- **Transpose**: Flipping a matrix over its diagonal. Rows become columns. Needed
  everywhere in attention computations.
- **Norms**: Measuring the "size" of a vector. L2 norm is the straight-line distance
  from the origin. Used in normalization.
- **Broadcasting**: How NumPy handles operations on arrays of different shapes.
  Essential for efficient implementation.

#### Calculus
- **Derivatives**: The rate of change. "If I nudge this input a tiny bit, how much
  does the output change?" This is all you need for ML.
- **Partial derivatives**: Same idea, but for functions with multiple inputs. "If I
  nudge just this ONE input, holding everything else fixed, how does the output
  change?"
- **The chain rule**: If y depends on x, and z depends on y, then the effect of x
  on z is the effect of x on y multiplied by the effect of y on z. This is
  BACKPROPAGATION. The entire field rests on this one rule.
- **Gradients**: A vector of all partial derivatives. Points in the direction of
  steepest increase. We go the opposite direction to minimize loss.

#### Probability and Statistics
- **Probability distributions**: A way of assigning likelihoods to outcomes. A
  language model is a probability distribution over the next word.
- **Softmax**: Takes any list of numbers and turns them into probabilities (positive,
  sum to 1). The exponential ensures all values are positive; the division by the
  sum ensures they add up to 1. This is how models express confidence.
- **Cross-entropy loss**: Measures how far the model's predicted distribution is from
  the true distribution. If the model assigns high probability to the correct answer,
  loss is low. If it assigns low probability, loss is high. This is the standard
  loss function for classification and language modeling.
- **Log probabilities**: We work in log space because multiplying many small
  probabilities causes underflow. Log turns products into sums.

#### Information Theory
- **Entropy**: The average "surprise" in a distribution. A fair coin has maximum
  entropy (most uncertain). A loaded coin has low entropy. Language models try to
  minimize the entropy of their predictions.
- **KL divergence**: Measures how different two probability distributions are. Not
  symmetric: KL(P||Q) != KL(Q||P). Shows up in variational methods and RLHF.
- **Perplexity**: 2^(cross-entropy). The "effective vocabulary size" the model is
  choosing from. A perplexity of 100 means the model is, on average, as uncertain
  as if it were choosing uniformly from 100 options. Lower is better.

### Papers to Read
- No formal papers for this phase. Use these references:
  - 3Blue1Brown, "Essence of Linear Algebra" (YouTube series) -- outstanding visual
    intuition
  - 3Blue1Brown, "Essence of Calculus" (YouTube series)
  - Chapter 2-4 of Goodfellow, Bengio, and Courville, "Deep Learning" (2016),
    freely available at https://www.deeplearningbook.org/

### Actionable Items

- [ ] **Implement vector and matrix operations from scratch in pure Python** (no NumPy)
  - Dot product of two lists
  - Matrix-vector multiplication
  - Matrix-matrix multiplication
  - Transpose
  - Verify results against NumPy

- [ ] **Implement the same operations using NumPy** and compare speed

- [ ] **Implement derivatives numerically**
  - Write a function `numerical_gradient(f, x, epsilon=1e-7)` that computes
    df/dx using (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
  - Test it on simple functions: f(x) = x^2, f(x) = sin(x)
  - Implement the chain rule: compute d/dx of sin(x^2) both analytically and
    numerically, verify they match

- [ ] **Implement softmax from scratch**
  - Naive version
  - Numerically stable version (subtract max before exponentiating)
  - Verify: outputs are positive and sum to 1
  - Plot softmax for different "temperatures" (divide inputs by T before softmax)

- [ ] **Implement cross-entropy loss from scratch**
  - For a single example
  - For a batch
  - Plot how loss changes as predicted probability of the correct class varies
    from 0 to 1

- [ ] **Implement entropy and KL divergence**
  - Compute entropy of various distributions
  - Show that KL divergence is zero when two distributions are identical
  - Show that KL divergence is not symmetric

### Blog Post
"The Math You Actually Need for Deep Learning (And Nothing More)" -- Cover the
essential operations with concrete examples, emphasizing intuition over formality.
Include interactive-style examples showing what dot products, softmax, and
cross-entropy actually compute on real numbers.

---

## Phase 1: Neural Network Fundamentals

**Goal**: Understand how neural networks learn, from the single neuron to
multi-layer networks trained with backpropagation. Build a working neural
network from scratch using only NumPy.

**Prerequisites**: Phase 0

### Key Concepts

#### The Perceptron (1958)
- Frank Rosenblatt's perceptron: the simplest possible neural network
- A single neuron: weighted sum of inputs + bias, passed through a step function
- Can learn linearly separable patterns (AND, OR) but not XOR
- The perceptron convergence theorem: if a solution exists, the perceptron will
  find it
- Why it matters: this is where everything started. The limitations of the
  perceptron (documented in Minsky & Papert, 1969) caused the first "AI winter"

#### Multi-Layer Perceptrons (MLPs)
- Stacking layers of neurons to learn non-linear patterns
- Hidden layers create intermediate representations
- Universal approximation theorem: with enough neurons in one hidden layer, an
  MLP can approximate any continuous function (Cybenko, 1989)
- XOR is now solvable: the hidden layer creates a new representation where XOR
  becomes linearly separable

#### Activation Functions
- **Step function**: Original perceptron. Not differentiable (cannot do gradient
  descent).
- **Sigmoid**: Smooth S-curve, outputs between 0 and 1. Differentiable everywhere.
  Problem: gradients vanish for large or small inputs ("saturating").
- **Tanh**: Like sigmoid but outputs between -1 and 1. Still saturates.
- **ReLU**: max(0, x). Simple, fast, no saturation for positive inputs. But "dead
  neurons" for negative inputs. The default choice in modern networks.
- **GELU**: Smooth approximation of ReLU. Used in GPT-2 and later transformers.

#### Backpropagation
- The algorithm that makes training deep networks possible
- Forward pass: compute output from input
- Loss computation: compare output to desired answer
- Backward pass: compute gradient of loss with respect to every weight using
  the chain rule
- The gradient tells each weight: "here is how you should change to reduce the
  loss"
- Key insight: backprop is just the chain rule applied systematically through
  the network, from output back to input

#### Gradient Descent and Optimization
- **SGD (Stochastic Gradient Descent)**: Update weights by stepping opposite to
  the gradient. "Stochastic" because we use mini-batches, not the full dataset.
- **Learning rate**: How big each step is. Too large: overshoot and diverge. Too
  small: converge painfully slowly.
- **Momentum**: "Keep rolling in the direction you've been going." Smooths out
  noisy gradients.
- **Adam**: Adaptive learning rates per parameter. The default optimizer in
  modern deep learning. Combines momentum with per-parameter scaling.

#### Loss Functions
- **Mean Squared Error (MSE)**: For regression. Average of (prediction - target)^2.
- **Binary Cross-Entropy**: For binary classification.
- **Categorical Cross-Entropy**: For multi-class classification. This is what
  language models use.

### Papers to Read
1. Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information
   Storage and Organization in the Brain." *Psychological Review*, 65(6), 386-408.
   -- Read for historical context; the math is accessible.

2. Rumelhart, D.E., Hinton, G.E., & Williams, R.J. (1986). "Learning
   representations by back-propagating errors." *Nature*, 323, 533-536.
   https://www.nature.com/articles/323533a0
   -- The foundational backpropagation paper. Short and readable.

3. Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training
   deep feedforward neural networks." *AISTATS*.
   http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
   -- Explains why weight initialization matters. Introduces Xavier initialization.

### Actionable Items

- [ ] **Build a Perceptron from scratch**
  - Implement the perceptron learning algorithm
  - Train it on AND, OR gates
  - Show it fails on XOR
  - Visualize the decision boundary

- [ ] **Build a 2-layer MLP from scratch**
  - Forward pass with sigmoid activation
  - Compute cross-entropy loss
  - Implement backpropagation BY HAND (derive all the gradients on paper first)
  - Train on XOR -- show it now works
  - Print weights at each epoch; watch them change

- [ ] **Implement all activation functions and their derivatives**
  - sigmoid, tanh, ReLU, GELU
  - Plot each one and its derivative
  - Verify derivatives numerically using the gradient checker from Phase 0

- [ ] **Build a configurable MLP class**
  - Arbitrary number of layers and neurons per layer
  - Choice of activation function
  - Forward and backward pass
  - Train on MNIST (load with simple Python, no sklearn)
  - Achieve >90% accuracy on MNIST with a 2-hidden-layer MLP

- [ ] **Implement and compare optimizers**
  - Plain SGD
  - SGD with momentum
  - Adam
  - Plot loss curves for each on the same problem
  - Observe how Adam converges faster

- [ ] **Gradient checking**
  - For every backprop implementation, verify against numerical gradients
  - This catches bugs that would otherwise be invisible

### Blog Post
"Neural Networks from Scratch: From a Single Neuron to Recognizing Digits" --
Tell the story chronologically: perceptron, its limitations, how stacking layers
and backpropagation solved the problem. Include the XOR visualization and MNIST
results.

---

## Phase 2: Sequence Models and the Path to Attention

**Goal**: Understand why sequences need special architectures, build RNNs and
LSTMs from scratch, and experience their limitations firsthand. These limitations
motivate everything that follows.

**Prerequisites**: Phase 1

### Key Concepts

#### Why Sequences Are Different
- Standard MLPs have a fixed input size. But text, speech, music, and time series
  have variable length.
- Order matters: "dog bites man" vs "man bites dog" have the same words but
  different meanings.
- MLPs have no concept of position or order. We need architectures that process
  sequences step by step.

#### Recurrent Neural Networks (RNNs)
- Process one element at a time, maintaining a "hidden state" that summarizes
  everything seen so far
- At each step: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
- The hidden state is a "memory" of the past
- Parameter sharing: the same weights are used at every time step
- Can handle variable-length sequences
- Trained with Backpropagation Through Time (BPTT): unroll the loop and apply
  standard backprop

#### The Vanishing Gradient Problem
- When backpropagating through many time steps, gradients get multiplied
  repeatedly by the weight matrix
- If the largest eigenvalue of W_hh is < 1, gradients shrink exponentially
  (vanish)
- If > 1, gradients grow exponentially (explode)
- Consequence: RNNs cannot learn long-range dependencies. They effectively
  "forget" what happened more than ~10-20 steps ago.
- This is not a theoretical concern -- you will observe it directly when you
  build one.

#### Long Short-Term Memory (LSTM)
- Hochreiter and Schmidhuber's solution (1997) to the vanishing gradient problem
- Key idea: add a "cell state" -- a highway that carries information across time
  steps with minimal interference
- Three gates control information flow:
  - **Forget gate**: What to erase from the cell state
  - **Input gate**: What new information to write to the cell state
  - **Output gate**: What to read from the cell state as the hidden state
- Gates use sigmoid (values 0-1), acting as soft switches
- Gradients can flow through the cell state without being multiplied by weight
  matrices at every step, solving the vanishing gradient problem

#### Gated Recurrent Unit (GRU)
- Cho et al. (2014): a simplified version of LSTM
- Two gates instead of three: reset gate and update gate
- No separate cell state
- Comparable performance to LSTM with fewer parameters

#### Sequence-to-Sequence (Seq2Seq)
- Sutskever et al. (2014): the architecture that made neural machine translation work
- Encoder: an RNN/LSTM that reads the input sequence and compresses it into a
  single vector (the "context vector")
- Decoder: an RNN/LSTM that generates the output sequence from the context vector
- The bottleneck problem: the entire input must be compressed into one fixed-size
  vector. For long sequences, this loses information.
- This bottleneck directly motivates the invention of attention (Phase 4).

### Papers to Read
1. Elman, J.L. (1990). "Finding Structure in Time." *Cognitive Science*, 14(2),
   179-211. https://doi.org/10.1207/s15516709cog1402_1
   -- The foundational RNN paper.

2. Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural
   Computation*, 9(8), 1735-1780.
   https://www.bioinf.jku.at/publications/older/2604.pdf
   -- The original LSTM paper. Dense but worth reading.

3. Cho, K., et al. (2014). "Learning Phrase Representations using RNN
   Encoder-Decoder for Statistical Machine Translation." *EMNLP*.
   https://arxiv.org/abs/1406.1078
   -- Introduces the GRU and the encoder-decoder framework.

4. Sutskever, I., Vinyals, O., & Le, Q.V. (2014). "Sequence to Sequence Learning
   with Neural Networks." *NeurIPS*. https://arxiv.org/abs/1409.3215
   -- The Seq2Seq paper. Clear writing, elegant architecture.

5. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). "On the difficulty of training
   Recurrent Neural Networks." *ICML*. https://arxiv.org/abs/1211.5063
   -- Formalizes the vanishing/exploding gradient problem in RNNs.

### Actionable Items

- [ ] **Build a vanilla RNN from scratch in NumPy**
  - Forward pass: implement the recurrence h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
  - Backward pass: implement BPTT (backpropagation through time)
  - Train on a character-level language model (predict next character)
  - Start with a tiny text (first few paragraphs of a book)
  - Sample text from the trained model

- [ ] **Demonstrate the vanishing gradient problem**
  - Train the RNN on sequences of increasing length
  - Plot gradient magnitudes at each time step
  - Show that gradients for early time steps become negligibly small

- [ ] **Build an LSTM from scratch in NumPy**
  - Implement all four equations (forget gate, input gate, cell update, output gate)
  - Implement BPTT through the LSTM
  - Train on the same character-level task
  - Show it handles longer sequences better than vanilla RNN
  - Plot gradient flow and compare to vanilla RNN

- [ ] **Build a GRU from scratch in NumPy**
  - Implement the two-gate architecture
  - Compare performance, training speed, and parameter count vs LSTM

- [ ] **Build a Seq2Seq model**
  - Encoder LSTM reads input, produces context vector
  - Decoder LSTM generates output from context vector
  - Train on a simple task: reversing a sequence of digits
  - Observe the bottleneck: performance degrades for longer sequences

### Blog Post
"Why Transformers Exist: The Journey Through RNNs and LSTMs" -- Tell the story
of sequence modeling as a progression of solutions to real problems. End with:
"what if the decoder could look back at the entire input instead of one vector?"

---

## Phase 3: Text Representation

**Goal**: Understand how text becomes numbers that neural networks can process.
Build tokenizers and embedding layers from scratch.

**Prerequisites**: Phase 1. Can be done in parallel with Phase 2.

### Key Concepts

#### Tokenization
- Neural networks work with numbers, not text. Tokenization converts text to numbers.
- **Character-level**: Each character is a token. Small vocabulary (~100), but
  sequences are very long.
- **Word-level**: Each word is a token. Captures meaning directly, but vocabulary
  is enormous (100K+) and cannot handle unseen words.
- **Subword (BPE -- Byte Pair Encoding)**: The sweet spot. Start with characters,
  merge the most frequent pairs iteratively. Handles any word, balances vocabulary
  size (~30K-50K) and sequence length. This is what GPT uses.

#### One-Hot Encoding
- Represent each token as a vector with a 1 at the token's index and 0s elsewhere
- Simple but wasteful: no similarity information ("cat" and "dog" are as far apart
  as "cat" and "algebra")

#### Word Embeddings
- Replace one-hot vectors with dense, learned vectors of dimension d (64-768)
- Words with similar meanings end up nearby in embedding space
- The embedding matrix E has shape (V, d): row i is the embedding for token i
- Looking up an embedding is just indexing into this matrix

#### Word2Vec
- Mikolov et al. (2013): learn embeddings by predicting context
- **Skip-gram**: Given a word, predict surrounding words
- **CBOW**: Given surrounding words, predict center word
- Famous: vec("king") - vec("man") + vec("woman") ~ vec("queen")

#### GloVe
- Pennington et al. (2014): learn embeddings from global co-occurrence statistics
- Combines count-based and prediction-based methods

#### Positional Encodings
- Transformers process all positions simultaneously, so they need position info
- Sinusoidal functions of different frequencies:
  PE(i, 2j) = sin(i / 10000^(2j/d)), PE(i, 2j+1) = cos(i / 10000^(2j/d))
- Allows the model to learn relative positions

### Papers to Read
1. Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in
   Vector Space." *ICLR Workshop*. https://arxiv.org/abs/1301.3781

2. Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases
   and their Compositionality." *NeurIPS*. https://arxiv.org/abs/1310.4546

3. Pennington, J., Socher, R., & Manning, C. (2014). "GloVe: Global Vectors for
   Word Representation." *EMNLP*. https://aclanthology.org/D14-1162/

4. Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of
   Rare Words with Subword Units." *ACL*. https://arxiv.org/abs/1508.07909
   -- Introduces BPE for NLP. The tokenization method GPT uses.

### Actionable Items

- [ ] **Build a character-level tokenizer**
  - encode() and decode() functions
  - Build vocabulary from a text corpus

- [ ] **Build a word-level tokenizer**
  - Splitting, lowercasing, handling punctuation
  - Build vocabulary with frequency thresholds
  - Handle unknown words (<UNK> token)

- [ ] **Implement Byte Pair Encoding (BPE) from scratch**
  - Start with character-level tokens
  - Count pair frequencies, merge the most frequent pair, repeat
  - Implement encode() and decode()
  - Compare vocabulary size and sequence lengths across all three methods

- [ ] **Implement an embedding layer from scratch**
  - Initialize random embedding matrix
  - Look up embeddings by index
  - Show that one-hot times embedding matrix equals index lookup
  - Implement gradient computation for the embedding matrix

- [ ] **Implement Word2Vec (Skip-gram with negative sampling)**
  - Train on a small text corpus
  - Visualize embeddings using PCA or t-SNE
  - Test word analogies

- [ ] **Implement sinusoidal positional encodings**
  - Generate the full positional encoding matrix
  - Visualize as a heatmap
  - Show that dot product between position vectors encodes relative distance

### Blog Post
"From Words to Vectors: How Machines Read Text" -- Cover the progression from
one-hot to embeddings, explain BPE, and end with positional encodings as the
bridge to transformers.

---

## Phase 4: Attention Mechanism

**Goal**: Understand attention as the key innovation that led to transformers.
Build every variant from scratch, culminating in multi-head self-attention.

**Prerequisites**: Phase 2 (Seq2Seq and the bottleneck), Phase 3 (embeddings)

### Key Concepts

#### The Bottleneck That Started It All
- Seq2Seq compresses the entire input into one vector
- Attention says: "instead of one vector, let the decoder look at ALL encoder
  hidden states and choose which ones to focus on"

#### Bahdanau Attention (Additive Attention, 2014)
- At each decoder step, compute an "alignment score" between the decoder state
  and every encoder state
- Scores go through softmax to become attention weights
- Context vector is a weighted sum of encoder states
- Alignment function: score(s_t, h_i) = v^T * tanh(W_1 * s_t + W_2 * h_i)

#### Luong Attention (Multiplicative Attention, 2015)
- Simpler: score(s_t, h_i) = s_t^T * W * h_i
- Faster to compute than Bahdanau
- Direct precursor to transformer attention

#### Self-Attention
- Instead of decoder-to-encoder, each position attends to all positions in the
  SAME sequence
- "The cat sat on the mat because it was tired" -- self-attention lets "it"
  attend strongly to "cat"

#### Scaled Dot-Product Attention (Transformer's Core)
- Three inputs: Query (Q), Key (K), Value (V)
- Analogy: Q is a search query, K is labels on filing cabinet drawers, V is the
  content inside. The query checks which keys match, then retrieves weighted values.
- Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
- Scale by sqrt(d_k) to prevent softmax saturation in high dimensions

#### Multi-Head Attention
- Run several attention functions in parallel with different learned projections
- Each head learns different relationship types (syntax, semantics, proximity)
- MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O

### Papers to Read
1. Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by
   Jointly Learning to Align and Translate." *ICLR*.
   https://arxiv.org/abs/1409.0473
   -- THE attention paper. Essential reading.

2. Luong, M., Pham, H., & Manning, C. (2015). "Effective Approaches to
   Attention-based Neural Machine Translation." *EMNLP*.
   https://arxiv.org/abs/1508.04025

3. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
   https://arxiv.org/abs/1706.03762
   -- Read Sections 3.2 and 3.2.2 now. Save the full paper for Phase 5.

### Actionable Items

- [ ] **Build Bahdanau attention from scratch**
  - Implement the additive alignment function
  - Integrate with Seq2Seq from Phase 2
  - Visualize attention weights as a heatmap
  - Compare performance with and without attention on longer sequences

- [ ] **Build Luong attention from scratch**
  - Implement dot, general, and concat variants
  - Compare against Bahdanau

- [ ] **Build scaled dot-product attention from scratch**
  - Implement Q, K, V formulation
  - Demonstrate the effect of scaling
  - Verify shapes at every step

- [ ] **Build self-attention from scratch**
  - Compute Q, K, V by projecting input with learned weight matrices
  - Apply scaled dot-product attention
  - Visualize which positions attend to which

- [ ] **Build multi-head attention from scratch**
  - Multiple parallel heads with separate projections
  - Concatenate and project
  - Visualize different heads learning different patterns

- [ ] **Attention masking**
  - Implement padding mask
  - Implement causal mask (each position only attends to earlier positions)

### Blog Post
"Attention: The Mechanism That Changed Everything" -- Start with the Seq2Seq
bottleneck, trace evolution to multi-head self-attention. Include attention heatmaps.

---

## Phase 5: The Transformer ("Attention Is All You Need")

**Goal**: Implement the full transformer architecture from scratch in NumPy.
Every component has been built in previous phases; now they come together.

**Prerequisites**: All of Phases 0-4

### Key Concepts

#### The Big Picture
- Vaswani et al. (2017): you do not need recurrence at all. Attention alone suffices.
- Processes the entire sequence in parallel (not step by step like RNNs)
- Two components: encoder and decoder. For GPT, only the decoder is used.

#### Encoder Architecture
- Input: token embeddings + positional encodings
- Stack of N identical layers, each with:
  1. Multi-head self-attention
  2. Feed-forward network (two linear layers with activation)
- Residual connections + layer normalization around each sub-layer

#### Decoder Architecture
- Two attention layers per block:
  1. **Masked self-attention**: causal mask prevents seeing the future
  2. **Cross-attention**: Q from decoder, K/V from encoder
  3. Feed-forward network
- Residual connections and layer normalization around each sub-layer

#### Layer Normalization
- Normalize across the feature dimension (not batch)
- Subtract mean, divide by std, then learned scale and shift
- Pre-norm (before sublayer) vs post-norm (after): pre-norm is more common now

#### Residual Connections
- output = x + Sublayer(x)
- Identity path lets gradients flow directly, enabling very deep networks

#### Feed-Forward Network (FFN)
- FFN(x) = W_2 * activation(W_1 * x + b_1) + b_2
- Inner dimension typically 4x model dimension
- Applied independently per position
- Original: ReLU. GPT-2: GELU.

#### Masking
- **Padding mask**: Ignore padding tokens in attention
- **Causal mask**: Upper triangular -infinity. Makes autoregressive generation possible.

### Papers to Read
1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
   https://arxiv.org/abs/1706.03762
   -- NOW read the full paper. Nothing should feel unfamiliar.

2. Ba, J.L., Kiros, J.R., & Hinton, G.E. (2016). "Layer Normalization."
   https://arxiv.org/abs/1607.06450

3. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
   https://arxiv.org/abs/1512.03385
   -- Where residual connections come from.

4. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer
   Architecture." *ICML*. https://arxiv.org/abs/2002.04745
   -- Pre-norm vs post-norm.

### Actionable Items

- [ ] **Implement layer normalization from scratch**
  - Forward: normalize, scale, shift
  - Backward: gradients for scale, shift, and input
  - Verify against known values

- [ ] **Assemble the transformer encoder block**
  - Multi-head self-attention + Add & Norm + FFN + Add & Norm
  - Stack N blocks

- [ ] **Assemble the transformer decoder block**
  - Masked self-attention + Add & Norm + Cross-attention + Add & Norm + FFN + Add & Norm
  - Stack N blocks

- [ ] **Build the full encoder-decoder transformer**
  - Embedding + positional encoding -> encoder -> decoder -> linear -> softmax
  - Run on a tiny sequence, verify shapes at every layer

- [ ] **Train on a toy translation task**
  - Reverse sequences, copy sequences, or digit-to-word translation
  - Full training loop with teacher forcing
  - Autoregressive decoding at inference time
  - This will be SLOW in NumPy. The point is understanding, not speed.

- [ ] **Ablation study**
  - Remove residual connections: observe instability
  - Remove layer norm: observe instability
  - Change number of heads: observe performance change
  - Change FFN dimension: observe performance change

### Blog Post
"Building a Transformer from Scratch in NumPy" -- Walk through the full
architecture. Include the paper's diagram annotated with your understanding.
Show training curves and outputs from the toy task.

---

## Phase 6: Transition to PyTorch

**Goal**: Learn PyTorch and rebuild the transformer. This should feel like
"automating what you already know."

**Prerequisites**: Phase 5

### Key Concepts

#### PyTorch Fundamentals
- **Tensors**: NumPy arrays with GPU support and automatic differentiation
- **Autograd**: Tracks operations, computes gradients automatically
- **nn.Module**: Base class for network components. Define in __init__, compute in forward()
- **nn.Parameter**: Tensors registered as learnable parameters
- **Optimizers**: torch.optim.Adam, SGD -- same algorithms, built in
- **DataLoader**: Batching, shuffling, parallel loading

#### MPS on M4 Max
- `device = torch.device("mps")` for GPU
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops
- Monitor: `torch.mps.current_allocated_memory()`
- Unified memory: GPU accesses all 64GB

### Papers to Read
- No papers. Use PyTorch docs: https://pytorch.org/tutorials/

### Actionable Items

- [ ] **PyTorch basics**
  - Create tensors, move to MPS
  - Use autograd, compare to hand-written backprop

- [ ] **Rebuild each component in PyTorch**
  - Embedding, positional encoding, attention, FFN, LayerNorm, full transformer
  - Verify each against NumPy version (same input -> compare outputs)

- [ ] **Training infrastructure**
  - Training loop, DataLoader, LR scheduling (warmup + cosine decay)
  - Gradient clipping, loss logging

- [ ] **Benchmark MPS vs CPU**
  - Time per batch, throughput comparison

### Blog Post
"From NumPy to PyTorch: What You Gain When a Framework Does the Math For You"

---

## Phase 7: GPT Architecture

**Goal**: Understand the GPT family -- decoder-only transformers. Build a GPT
model in PyTorch.

**Prerequisites**: Phase 6

### Key Concepts

#### From Transformer to GPT
- GPT uses ONLY the decoder (with causal masking)
- No cross-attention. Each block has masked self-attention + FFN.
- Simpler than the full transformer.

#### GPT-1 (2018)
- Radford et al. at OpenAI
- Unsupervised pre-training, then supervised fine-tuning
- 12 layers, 12 heads, 768 dim (~117M params)
- Trained on BooksCorpus (~800M words)

#### GPT-2 (2019)
- Same architecture, bigger. Four sizes: 124M, 355M, 774M, 1.5B
- Trained on WebText (~40GB)
- Key: zero-shot task performance
- GPT-2 Small (124M) is our training target

#### GPT-3 (2020)
- 175B parameters
- In-context learning: learns from examples in the prompt
- We will not train this size, but the scaling insight matters

#### Key Architectural Details
- Learned positional embeddings (not sinusoidal)
- Pre-norm (LayerNorm before attention/FFN)
- GELU activation in FFN
- Weight tying: input embeddings = output projection matrix
- BPE vocabulary: ~50,257 tokens
- Context length: 1024

### Papers to Read
1. Radford, A., et al. (2018). "Improving Language Understanding by Generative
   Pre-Training." (GPT-1)
   https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

2. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask
   Learners." (GPT-2)
   https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." (GPT-3)
   *NeurIPS*. https://arxiv.org/abs/2005.14165

### Actionable Items

- [ ] **Implement GPT model in PyTorch**
  - Start small: 4 layers, 4 heads, 128 dim (~1M params)
  - Causal self-attention, pre-norm, GELU, learned positional embeddings
  - Weight tying

- [ ] **Train on Shakespeare**
  - Monitor loss, generate samples every N steps
  - Watch progression from gibberish to coherent text

- [ ] **Scale progressively**: 1M -> 10M -> 50M params
  - Observe training speed, loss curves, generation quality at each scale

- [ ] **Implement text generation**
  - Greedy, temperature sampling, top-k, top-p (nucleus)
  - Compare: greedy is repetitive, high temp is chaotic, top-p is balanced

- [ ] **Compare to nanoGPT**
  - Read Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT)
  - Compare architecture choices, identify differences

### Blog Post
"GPT Demystified: How Decoder-Only Transformers Generate Text" -- Walk through
the architecture, pre-training objective, and generation strategies. Show outputs
at different scales and temperatures.

---

## Phase 8: Training Your Own Small GPT

**Goal**: Train a GPT-2 Small (~124M params) on real data using M4 Max.
This is the main deliverable.

**Prerequisites**: Phase 7

### Key Concepts

#### Dataset Options
- **OpenWebText**: Open-source GPT-2 training data reproduction. ~38GB.
- **FineWeb-Edu**: HuggingFace educational web content. High quality.
- **TinyStories**: Simple stories for smaller models / faster iteration.

#### Training Configuration for M4 Max 64GB

**Model (GPT-2 Small)**:
- 12 layers, 12 heads, d_model=768, d_ff=3072
- Context: 1024, vocab: 50257, total: ~124M params

**Hyperparameters**:
- Batch: 8-16, gradient accumulation: 8-16 (effective: 64-256)
- LR: 6e-4 peak, warmup 2000 steps, cosine decay to 6e-5
- Weight decay: 0.1, Adam betas: (0.9, 0.95), grad clip: 1.0
- Target: ~2-10B tokens

**Memory Budget**:
- Model: ~0.5GB, optimizer: ~1GB, gradients: ~0.5GB, activations: ~4-8GB
- Total: ~6-10GB, well within 64GB

**Time Estimates**:
- ~10K-30K tokens/sec on M4 Max
- 2B tokens: ~1-2 days. 10B tokens: ~4-6 days.

### Papers to Read
1. Karpathy, A. (2023). "Let's build GPT" (YouTube + code).
   https://github.com/karpathy/build-nanogpt
   -- Watch AFTER building your own.

2. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models."
   (Chinchilla). https://arxiv.org/abs/2203.15556
   -- Optimal model-size vs data-size balance.

### Actionable Items

- [ ] **Prepare training data**
  - Download OpenWebText or FineWeb-Edu subset (~10GB)
  - Tokenize with tiktoken or your BPE
  - Store as memory-mapped binary files

- [ ] **Finalize model**
  - Scale to 124M params
  - Weight init (normal std=0.02, residual scaling 1/sqrt(n_layers))
  - Dropout (0.1 train, 0 eval)

- [ ] **Full training loop**
  - Gradient accumulation, LR warmup + cosine decay
  - Gradient clipping, periodic eval, checkpointing, resume capability

- [ ] **Train the model**
  - Start small (100M tokens, ~1 hour) to verify
  - Scale to 2-10B tokens
  - Monitor loss, perplexity, gradient norms
  - Generate samples at regular intervals

- [ ] **Evaluate and generate**
  - Perplexity on held-out test set
  - Generate with different prompts and sampling strategies
  - Honestly assess: 124M on 2B tokens = coherent but not brilliant

- [ ] **Ablation experiments**
  - Context lengths (256, 512, 1024)
  - Learning rates, warmup vs no warmup
  - Different data amounts

### Blog Post
"Training GPT-2 Small on a MacBook: A Practical Guide" -- Full process: data
prep, hyperparameters, training dynamics, results. Loss curves, sample outputs,
honest assessment. Apple Silicon specifics.

---

## Phase 9: Modern Developments and State of the Art

**Goal**: Survey major advances since GPT-2/3. Understand what makes modern
LLMs better than vanilla transformers.

**Prerequisites**: Phase 8

### Key Concepts

#### Scaling Laws
- Kaplan et al. (2020): performance improves as power law of size, data, compute
- Chinchilla (2022): prior models were undertrained. Balance model size and data.

#### RLHF (Reinforcement Learning from Human Feedback)
- Pre-training gives knowledge. RLHF aligns with human preferences.
- SFT -> Reward model -> PPO optimization
- This is how ChatGPT was made helpful

#### DPO (Direct Preference Optimization)
- Rafailov et al. (2023): skip reward model, optimize directly from preferences
- Simpler and often more stable than PPO

#### Flash Attention
- Dao et al. (2022): exact attention, O(N) memory instead of O(N^2)
- Tiles computation to stay in fast SRAM

#### Rotary Position Embeddings (RoPE)
- Su et al. (2021): encode relative position by rotating Q and K vectors
- Generalizes to longer sequences than training
- Used in LLaMA, Mistral, and most modern models

#### KV-Cache
- Cache K and V from previous steps during generation
- Reduces generation from O(n^2) to O(n)

#### Grouped-Query Attention (GQA)
- Share K, V across groups of heads
- Reduces KV-Cache memory with minimal quality loss

#### Mixture of Experts (MoE)
- Route each token to a subset of expert FFN layers
- Many parameters, constant computation per token

#### Modern Architectures
- **LLaMA**: RoPE, RMSNorm, SwiGLU, no bias. Open-source.
- **Mistral**: Sliding window attention, GQA. 7B competing with much larger models.
- **Mixtral**: Mistral + MoE. 46.7B total, 12.9B active per token.

### Papers to Read
1. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models."
   https://arxiv.org/abs/2001.08361

2. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models."
   https://arxiv.org/abs/2203.15556

3. Ouyang, L., et al. (2022). "Training language models to follow instructions
   with human feedback." https://arxiv.org/abs/2203.02155

4. Rafailov, R., et al. (2023). "Direct Preference Optimization."
   https://arxiv.org/abs/2305.18290

5. Dao, T., et al. (2022). "FlashAttention." https://arxiv.org/abs/2205.14135

6. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
   Embedding." https://arxiv.org/abs/2104.09864

7. Touvron, H., et al. (2023). "LLaMA." https://arxiv.org/abs/2302.13971

8. Jiang, A., et al. (2023). "Mistral 7B." https://arxiv.org/abs/2310.06825

9. Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need."
   https://arxiv.org/abs/1911.02150

10. Ainslie, J., et al. (2023). "GQA." https://arxiv.org/abs/2305.13245

### Actionable Items

- [ ] **Implement RoPE**: Replace learned positional embeddings, test on different lengths
- [ ] **Implement KV-Cache**: Benchmark generation speed with and without
- [ ] **Implement RMSNorm**: Drop-in LayerNorm replacement
- [ ] **Implement SwiGLU activation**: Replace GELU in FFN
- [ ] **Read and annotate papers**: 1-page summary for each (problem, solution, results)
- [ ] **Optional: Implement simplified DPO pipeline**

### Blog Post
"From GPT-2 to LLaMA: What Changed and Why" -- Survey key innovations. Explain
each in terms of the problem it solves. Modern LLMs are careful engineering on
the same transformer foundation.

---

## Phase 10: Blog Writing and Documentation

**Goal**: Consolidate everything into a series of blog posts that could teach
someone else the same path.

**Prerequisites**: All previous phases (write drafts as you go)

### Actionable Items

- [ ] **Finalize all phase blog posts** (one per phase minimum)
- [ ] **Create diagrams and visualizations** (architecture diagrams, attention
  heatmaps, loss curves, embedding visualizations)
- [ ] **Write capstone post**: "Everything I Learned Building a GPT From Scratch"
- [ ] **Review and edit**: Check code snippets, ensure accessibility

### Complete Blog Post List

| Phase | Topic |
|-------|-------|
| 0 | The Math You Actually Need for Deep Learning |
| 1 | Neural Networks from Scratch: From a Single Neuron to Recognizing Digits |
| 2 | Why Transformers Exist: The Journey Through RNNs and LSTMs |
| 3 | From Words to Vectors: How Machines Read Text |
| 4 | Attention: The Mechanism That Changed Everything |
| 5 | Building a Transformer from Scratch in NumPy |
| 6 | From NumPy to PyTorch: What You Gain When a Framework Does the Math |
| 7 | GPT Demystified: How Decoder-Only Transformers Generate Text |
| 8 | Training GPT-2 Small on a MacBook: A Practical Guide |
| 9 | From GPT-2 to LLaMA: What Changed and Why |
| 10 | Everything I Learned Building a GPT From Scratch |

---

## Complete Paper Reading List

### Foundations
| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 1 | Rosenblatt, "The Perceptron" | 1958 | Where neural networks began |
| 2 | Rumelhart et al., "Learning representations by back-propagating errors" | 1986 | Backpropagation |
| 3 | Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" | 2010 | Weight initialization |

### Sequence Models
| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 4 | Elman, "Finding Structure in Time" | 1990 | Foundational RNN |
| 5 | Hochreiter & Schmidhuber, "Long Short-Term Memory" | 1997 | LSTM |
| 6 | Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" | 2014 | GRU and encoder-decoder |
| 7 | Sutskever et al., "Sequence to Sequence Learning with Neural Networks" | 2014 | Seq2Seq |
| 8 | Pascanu et al., "On the difficulty of training Recurrent Neural Networks" | 2013 | Vanishing gradients |

### Text Representation
| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 9 | Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" | 2013 | Word2Vec |
| 10 | Mikolov et al., "Distributed Representations of Words and Phrases" | 2013 | Negative sampling |
| 11 | Pennington et al., "GloVe" | 2014 | GloVe embeddings |
| 12 | Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" | 2016 | BPE tokenization |

### Attention and Transformers
| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 13 | Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" | 2015 | Original attention |
| 14 | Luong et al., "Effective Approaches to Attention-based Neural Machine Translation" | 2015 | Simplified attention |
| 15 | Vaswani et al., "Attention Is All You Need" | 2017 | THE transformer paper |
| 16 | Ba et al., "Layer Normalization" | 2016 | Layer normalization |
| 17 | He et al., "Deep Residual Learning for Image Recognition" | 2016 | Residual connections |
| 18 | Xiong et al., "On Layer Normalization in the Transformer Architecture" | 2020 | Pre-norm vs post-norm |

### GPT Family
| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 19 | Radford et al., GPT-1 | 2018 | Pre-train + fine-tune |
| 20 | Radford et al., GPT-2 | 2019 | Zero-shot transfer |
| 21 | Brown et al., GPT-3 | 2020 | In-context learning, scaling |

### Modern Advances
| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 22 | Kaplan et al., "Scaling Laws for Neural Language Models" | 2020 | Predictable scaling |
| 23 | Hoffmann et al., "Chinchilla" | 2022 | Optimal model/data balance |
| 24 | Ouyang et al., "InstructGPT" | 2022 | RLHF |
| 25 | Rafailov et al., "DPO" | 2023 | Simpler alignment |
| 26 | Dao et al., "FlashAttention" | 2022 | Memory-efficient attention |
| 27 | Su et al., "RoFormer" (RoPE) | 2021 | Rotary position embeddings |
| 28 | Touvron et al., "LLaMA" | 2023 | Modern open LLM blueprint |
| 29 | Jiang et al., "Mistral 7B" | 2023 | Efficient innovations |
| 30 | Shazeer, "Multi-Query Attention" | 2019 | MQA |
| 31 | Ainslie et al., "GQA" | 2023 | Grouped-Query Attention |

### Bonus: Broader Context
| # | Paper | Year | Why Read It |
|---|-------|------|-------------|
| 32 | Bengio et al., "A Neural Probabilistic Language Model" | 2003 | First neural LM |
| 33 | Devlin et al., "BERT" | 2019 | Encoder-only (contrast to GPT) |
| 34 | Wei et al., "Chain-of-Thought Prompting" | 2022 | Emergent capabilities |
| 35 | Dao et al., "FlashAttention-2" | 2023 | Improved Flash Attention |

---

## Timeline Summary

| Phase | Topic | Est. Time | Cumulative |
|-------|-------|-----------|------------|
| 0 | Math Foundations | 1-2 weeks | 1-2 weeks |
| 1 | Neural Networks | 2-3 weeks | 3-5 weeks |
| 2 | Sequence Models | 2-3 weeks | 5-8 weeks |
| 3 | Text Representation | 1-2 weeks | 6-10 weeks |
| 4 | Attention | 2-3 weeks | 8-13 weeks |
| 5 | Transformer | 3-4 weeks | 11-17 weeks |
| 6 | PyTorch Transition | 1-2 weeks | 12-19 weeks |
| 7 | GPT Architecture | 2-3 weeks | 14-22 weeks |
| 8 | Training GPT | 3-4 weeks | 17-26 weeks |
| 9 | Modern Developments | 2-3 weeks | 19-29 weeks |
| 10 | Blog Writing | 2-3 weeks | 21-32 weeks |

**Total: ~5-8 months** at 10-20 hours/week. Faster full-time.

Note: Phase 3 can run in parallel with Phase 2.

---

## What You Will Build (Checklist)

- [ ] Vector and matrix operations (pure Python and NumPy)
- [ ] Gradient computation and numerical gradient checking
- [ ] Softmax, cross-entropy, entropy, KL divergence
- [ ] Perceptron
- [ ] Multi-layer perceptron with backpropagation
- [ ] All common activation functions and derivatives
- [ ] SGD, SGD with momentum, Adam optimizer
- [ ] Vanilla RNN with BPTT
- [ ] LSTM with BPTT
- [ ] GRU
- [ ] Seq2Seq encoder-decoder
- [ ] Character-level, word-level, and BPE tokenizers
- [ ] Embedding layer
- [ ] Word2Vec (skip-gram with negative sampling)
- [ ] Sinusoidal positional encodings
- [ ] Bahdanau attention
- [ ] Luong attention
- [ ] Scaled dot-product attention
- [ ] Multi-head attention
- [ ] Attention masking (padding and causal)
- [ ] Layer normalization
- [ ] Full transformer (encoder + decoder) in NumPy
- [ ] Full transformer in PyTorch
- [ ] GPT model (decoder-only) in PyTorch
- [ ] Text generation (temperature, top-k, top-p)
- [ ] Training pipeline for 124M parameter GPT on M4 Max
- [ ] RoPE, KV-Cache, RMSNorm, SwiGLU (modern improvements)
- [ ] 11 blog posts documenting the journey
