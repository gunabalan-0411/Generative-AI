# Generative AI Interview Notes — LLMs + GPT + Tokenization + Transformers

> Goal: interview-ready notes that are **easy to revise** and **easy to explain** as a Data Scientist.
> Scope (only): **LLMs, GPT architecture, how LLMs work, tokenization, custom tokenizers, Transformer breakthrough (Attention paper)**.

---

## 1) What are LLMs? (Large Language Models)

### Definition (simple)

An **LLM is a neural network trained to predict the next token** (word/part-word) from a large dataset.

### Key idea

LLMs don’t “memorize answers” directly. They learn:

* statistical patterns in language
* relationships between tokens
* how to generate coherent sequences

### What makes it “Large”?

* **parameters** (millions → billions → trillions)
* **training data** (internet-scale text)
* **compute** (GPUs/TPUs)

### What can LLMs do?

* text generation
* summarization
* classification
* Q&A
* translation
* code generation

✅ Interview line:

> “LLMs are next-token predictors trained at scale; intelligence emerges from scale + architecture + data.”

---

## 2) LLMs Work Under the Hood (Core Mechanics)

### The overall pipeline (high-level)

1. Text input
2. Tokenization
3. Tokens → embeddings
4. Transformer layers compute contextual representations
5. Output logits for vocabulary
6. Decode tokens into text

---

### Training objective: Next-token prediction

Given sequence:
`"I love machine"` → model predicts next token: `" learning"`

Mathematically:

* model learns probability:  (P(token_t | token_{<t}))

---

### How LLM learns during training

* forward pass: compute prediction
* loss: cross entropy (difference between predicted distribution & true next token)
* backpropagation: update weights
* repeated for huge data

---

### Inference (generation time)

At inference:

* model outputs probability distribution over vocabulary
* decoding strategy chooses the next token

Common decoding (interview-level):

* **Greedy decoding**: pick highest probability token
* **Beam search**: keep top-k sequences (good for translation)
* **Top-k sampling**: sample from top k tokens
* **Top-p (nucleus) sampling**: sample from smallest set with cumulative prob p
* **Temperature**: controls randomness

✅ Interview line:

> “Training is optimizing next-token probability; inference uses decoding strategies to generate sequences.”

---

## 3) Tokenization in NLP (must know)

### What is tokenization?

Tokenization converts raw text into **tokens** (numbers).

Example:
Text: `"ChatGPT is amazing"`
Tokens: `[1298, 445, 9081]` (example IDs)

### Why tokenization is needed?

Neural networks cannot directly process raw strings. They need integers.

---

### Types of tokenization

#### A) Word-level (old)

* tokens are full words
* problem: huge vocabulary + OOV (out of vocabulary)

#### B) Character-level

* tokens are individual letters
* problem: very long sequences

#### C) Subword tokenization (modern standard ✅)

Used by GPT/BERT-like models.

* tokens are chunks like `"un"`, `"happy"`, `"ness"`

Why subword is best?

* handles rare words
* smaller vocab than word-level
* shorter sequences than char-level

---

### Most common subword algorithms

* **BPE (Byte Pair Encoding)** → used in GPT family historically
* **WordPiece** → used in BERT
* **Unigram LM** → used in SentencePiece

✅ Interview line:

> “Modern LLMs use subword tokenization to handle rare words without exploding vocabulary size.”

---

## 4) GPT Architecture (Interview Core)

### What is GPT?

GPT = **Generative Pretrained Transformer**.

Key properties:

* **Transformer Decoder-only** architecture
* trained with **causal language modeling** (next-token prediction)

---

### GPT is NOT encoder-decoder

* BERT: encoder-only
* T5: encoder-decoder
* GPT: decoder-only (autoregressive)

✅ Interview line:

> “GPT is decoder-only: it generates tokens left-to-right using causal attention.”

---

### GPT model components

#### 1) Token Embedding

Maps token IDs → vectors.

#### 2) Positional Encoding / Positional Embeddings

Adds order information.
Why?

* attention alone has no sense of token order

#### 3) Transformer Blocks (repeated N times)

Each block typically has:

* LayerNorm
* **Masked Multi-Head Self Attention**
* Feed Forward Network (MLP)
* Residual connections

#### 4) Output projection

Final hidden states → logits over vocabulary.

---

### Masked (causal) attention

In GPT, token at position *t* can only attend to:

* tokens ≤ t

This prevents “peeking into the future”.

---

## 5) Transformer Breakthrough — Attention Is All You Need (Google Paper)

### Why it was a breakthrough?

Before Transformers:

* RNN/LSTM were used for sequences
* they process sequentially → slow + long-range dependency issues

Transformers:

* parallel processing
* strong long context learning
* faster training on GPUs

✅ Interview line:

> “Transformers replaced recurrence with attention, enabling parallel training and better long-range dependencies.”

---

### What is Attention (simple)

Attention means:

> each token decides how much to focus on other tokens.

In sentence:
`"The animal didn’t cross the road because it was tired"`
Attention helps resolve `it` = animal.

---

### Attention formula (interview safe)

* Query (Q), Key (K), Value (V)
* attention weights = similarity(Q, K)
* output = weighted sum of V

Scaled dot-product attention:
(Attention(Q,K,V) = softmax(QK^T/\sqrt{d_k})V)

---

### Multi-head attention

Instead of one attention:

* multiple heads learn different relations

Example heads learn:

* grammar
* coreference
* semantic similarity

---

## 6) Custom Tokenizer (Customer Tokenizer) — What & Why

### What is a custom tokenizer?

A tokenizer trained/customized for a **specific domain**:

* healthcare
* finance
* customer support chat
* product manuals

### Why do we need it?

Because general tokenizers may split domain words badly.
Example:

* `"HPDeskJet2734"` might become many useless tokens

With custom tokenizer:

* domain terms become stable tokens
* shorter sequences
* better modeling efficiency

---

### When to build a custom tokenizer (practical)

Do it when:

* domain has many product codes / acronyms
* lots of non-English / mixed language
* you are training/fine-tuning from scratch or large-scale domain adaptation

If only prompt engineering / small fine-tune:

* you usually keep original tokenizer

---

### How custom tokenization is built (high-level)

1. collect large domain corpus
2. choose algorithm (BPE / Unigram)
3. train tokenizer vocab size (e.g., 32k)
4. validate token splits
5. retrain if needed

✅ Interview line:

> “Custom tokenizers reduce fragmentation of domain terms, improving efficiency and representation quality.”

---

## 7) Quick Interview Answers (ready-to-speak)

### Q1: What is an LLM?

> “An LLM is a large neural network trained to predict the next token. With Transformers + huge data, it learns language patterns and can generate coherent text.”

### Q2: What is special about GPT architecture?

> “GPT is decoder-only Transformer with causal masking. It generates text autoregressively left-to-right.”

### Q3: Why Transformers replaced LSTMs?

> “Transformers use attention to model dependencies without sequential recurrence, enabling parallel training and better long-range context.”

### Q4: What is tokenization?

> “Tokenization converts text into subword tokens so the model can process them as integers. Subword tokenization balances vocabulary size and coverage.”

---

## 8) Mini Glossary

* **Token**: smallest unit the model reads/writes
* **Embedding**: vector representation of a token
* **Logits**: raw scores for each token in vocab
* **Softmax**: converts logits into probabilities
* **Causal Mask**: prevents attention to future tokens
* **Decoder-only**: generates autoregressively

# Generative AI Interview Notes — Embeddings + Positional Encoding + Encoder/Decoder + Autoencoders + Multi-Head Attention

> Goal: interview-ready **deep dive** notes, explained for a Data Scientist.
> Scope (only): vector embeddings, positional encoding, autoencoders, encoder vs decoder vs bidirectional, multi-head attention.

---

## 1) Deep dive into Vector Embeddings

### What is an embedding? (simple definition)

An **embedding** is a dense numeric vector that represents meaning.

* A word, sentence, document, image, or user can be converted into a vector.
* Similar meanings → vectors are close together.

✅ Interview line:

> “Embedding is a learned vector representation that captures semantic similarity.”

---

### Why embeddings matter in Generative AI

Embeddings are the backbone for:

* **semantic search**
* **RAG (retrieval augmented generation)**
* clustering of documents
* recommendations
* deduplication

Even though LLMs generate tokens, embeddings power:

> *finding the right context*.

---

### Types of embeddings

#### A) Token embeddings (inside LLM)

* Each token ID → vector
* used as input to Transformer

#### B) Sentence / Document embeddings (external)

* One vector for a sentence/document
* used for retrieval / similarity

Example models:

* sentence-transformers
* OpenAI text-embedding models

---

### How similarity is measured (must know)

#### Cosine similarity (most common)

* compares direction (angle) of vectors
* ignores magnitude

(cos(\theta) = \frac{A·B}{||A|| ||B||})

Range:

* +1 → same direction (very similar)
* 0 → unrelated
* -1 → opposite

✅ Interview line:

> “Vector DB retrieval typically uses cosine similarity between embeddings.”

---

### What controls embedding quality?

* training objective (contrastive learning, next-token, etc.)
* embedding dimension (e.g., 384, 768, 1536)
* domain alignment (general vs domain-specific)

---

### Embeddings in RAG (high-level concept)

Even though you didn’t ask RAG, embeddings are incomplete without this:

1. Convert documents → embeddings
2. Store in vector database
3. User query → query embedding
4. Retrieve top-k similar chunks
5. Send retrieved context + query into LLM

✅ Interview line:

> “Embeddings act like an index for meaning-based search; RAG uses them to fetch relevant context.”

---

## 2) Role of Positional Encoding in Transformers

### Problem: Attention has no order

A Transformer sees tokens as a set.
Without extra info:

* it cannot tell whether "dog bites man" vs "man bites dog"

So we need to inject position information.

---

### What is positional encoding?

A way to add **token order** into embeddings.

Input to Transformer:
(x_t = tokenEmbedding(token_t) + positionEmbedding(t))

---

### Two main types

#### A) Fixed positional encodings (original paper)

* uses sine/cosine functions
* no learning required

#### B) Learned positional embeddings (GPT style historically)

* position vectors are learned parameters

---

### Modern positional techniques (important for interviews)

#### Rotary Positional Embedding (RoPE)

* rotates Q/K vectors based on position
* improves long-context extrapolation

#### ALiBi

* adds linear bias to attention scores
* handles long context efficiently

✅ Interview line:

> “Positional encoding is critical because self-attention alone doesn’t know token order.”

---

## 3) What are Autoencoders?

### Definition

An **autoencoder** is a neural network trained to:

* compress input into a small representation (latent space)
* reconstruct input back

Structure:

* **Encoder:** compress
* **Decoder:** reconstruct

---

### Why autoencoders matter in GenAI?

They are used for:

* dimensionality reduction
* anomaly detection
* representation learning

And especially:

* **VAEs (Variational Autoencoders)** → generative models
* **Latent diffusion** in image generation (stable diffusion uses latent space)

✅ Interview line:

> “Autoencoders learn compressed latent representations; VAEs make them generative.”

---

### Simple explanation with example

Input: image 256×256

* encoder reduces to vector (latent)
* decoder reconstructs image

Latent captures: shape, style, features.

---

## 4) Encoder vs Decoder vs Bidirectional (Must know)

### Encoder (BERT-style)

* reads full input sequence
* can attend to past and future tokens
* used for: classification, extraction, embeddings

✅ key property: **bidirectional attention**

Example: BERT.

---

### Decoder (GPT-style)

* generates output left-to-right
* can only attend to past tokens
* used for: text generation

✅ key property: **causal / masked attention**

Example: GPT.

---

### Encoder-Decoder (T5 / Transformer original)

* encoder builds representation of input
* decoder generates output conditioned on encoder

Used for:

* translation
* summarization

Example: T5, original Transformer.

---

### Bidirectional meaning

Bidirectional means:

> token at position t can use both left + right context.

This is powerful for understanding tasks.

✅ Interview line:

> “Encoders are bidirectional (understanding). Decoders are causal (generation).”

---

## 5) Understanding Multi-Head Attention (MHA) for rich context

### What is self-attention?

Each token computes relevance with all other tokens and mixes information.

Example:
`"The trophy didn’t fit into the suitcase because it was too big"`
Attention helps link **it** → trophy.

---

### Why multi-head?

One attention head cannot capture all patterns.

Multi-head attention runs multiple attentions in parallel.
Each head learns different relationships:

* syntactic structure
* coreference
* positional dependency
* semantic similarity

Then heads are concatenated and projected.

---

### Intuition (interview-friendly)

* **Head 1:** grammar
* **Head 2:** meaning
* **Head 3:** subject/object relations

✅ Interview line:

> “Multi-head attention lets the model look at the same sentence from multiple perspectives.”

---

### Basic formula (safe level)

For each head i:

* compute (Q_i, K_i, V_i)
* (head_i = Attention(Q_i,K_i,V_i))

Then:
(MHA = Concat(head_1,...,head_h)W^O)

---

## 6) Extra Important Concepts You Should Know (strong interview value)

### 6.1 Latent space

* compressed feature space
* used in autoencoders, VAEs, diffusion

### 6.2 Attention vs MLP roles

* attention: mixes information across tokens
* MLP: transforms information per token

### 6.3 Training vs inference difference

* training uses teacher forcing (known next token)
* inference generates one token at a time

### 6.4 Context window

* max tokens model can attend to
* affects cost + memory + retrieval strategy

---

## 7) Quick Interview Answers (ready-to-speak)

### Q1: Why embeddings are useful?

> “They convert language into vectors so we can do semantic similarity, retrieval, clustering, and power RAG pipelines.”

### Q2: Why positional encoding is required?

> “Attention doesn’t understand order, so positional encoding injects token position into the model.”

### Q3: Encoder vs decoder?

> “Encoder reads full context bidirectionally for understanding; decoder generates text with causal masking.”

### Q4: Why multi-head attention?

> “Multiple heads learn different relationships in parallel, producing richer contextual understanding.”

---

## 8) Mini Glossary

* **Embedding**: vector representation of meaning
* **Latent space**: compressed representation
* **Cosine similarity**: similarity metric in vector search
* **Positional encoding**: adds order info
* **Bidirectional**: attends left + right
* **Causal mask**: blocks future tokens
* **MHA**: multi-head attention





