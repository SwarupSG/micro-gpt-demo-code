# %% [markdown]
# # microGPT — a language model in ~200 lines of pure Python
#
# ## What are we building?
#
# A tiny neural network that learns to invent plausible-sounding baby names — things like `kamon`, `jaire`, `areli` — after reading 32,000 real names from a file.
#
# It has never been told what a name *is*. It just sees letter sequences and tries to guess the next letter. After a thousand training steps, it has absorbed enough statistical patterns (vowels follow consonants, `ia` endings are common, names rarely double `z`, etc.) that its own outputs look name-like.
#
# ---
#
# ## The whole thing in three ideas
#
# 1. **Forward pass.** Feed a letter in → model predicts the next letter.
# 2. **Measure wrongness.** Compare the prediction to the real next letter → a single **loss** number.
# 3. **Nudge the knobs.** Adjust the model's internal numbers slightly so it would have been less wrong. Repeat.
#
# Everything below is machinery for those three steps. The code is dependency-free — no PyTorch, no NumPy — so every concept (autograd, attention, optimizer) is built by hand and you can read it top to bottom.
#
# ---
#
# ## How to read this notebook
#
# Each code cell is preceded by a markdown cell that gives the **mental model first** — the analogy and the *why* — before the mechanics. Read those first; the code cell is then a natural translation of the idea into Python.
#
# ---
#
# ## Prerequisites
#
# **What you need going in:**
#
# - **Basic Python.** Comfort with: lists and list comprehensions, slicing (`x[2:5]`), `for` loops, defining classes, and lambdas. We use a few magic methods (`__add__`, `__mul__`, `__repr__`) — if you haven't seen those, they're introduced inline.
# - **High-school math.** You should know what a *derivative* and a *matrix* are. We build up from there — no calculus or linear algebra fluency assumed.
# - **No ML background required.** Every ML concept (gradients, backprop, attention, softmax, cross-entropy, Adam) is built from scratch with analogies before the code uses it.
#
# **What you do *not* need:**
#
# - PyTorch, NumPy, or any ML framework. The whole notebook is ~100 lines of pure Python — no `pip install` beyond what ships with CPython.
# - A GPU. Training takes a few minutes on a laptop.
# - A prior understanding of transformers or LLMs — that's what you're building.
#
# **Estimated time to work through:** 60–90 minutes if you read every cell carefully and run the demos. Less if you skim; more if you experiment with the exercises at the end.
#
# ---
#
# Credits: architecture and algorithm are from @karpathy's `micrograd` / `nanoGPT` work.

# %% [markdown]
# ## Where microGPT fits in the full LLM pipeline
#
# Modern chat models like ChatGPT are built in **three stages**. This notebook only covers Stage 1 — but that framing matters, because every limitation you see in the babbling output at the end of this notebook is a direct consequence of skipping Stages 2 and 3.
#
# ### Stage 1 — Pre-training (the **base model**)
# Feed the model a giant pile of text. Let it predict the next token, billions of times. The result is a *"glorified autocomplete"* — an **internet document simulator** that can continue any prompt in a plausible style but does not answer questions. **This is what we build below.**
#
# ### Stage 2 — Supervised Fine-Tuning (the **assistant persona**)
# Show the base model ~100k hand-written *(prompt, ideal reply)* pairs. It learns to imitate the tone of a helpful human labeler. *Skipped in this notebook.*
#
# ### Stage 3 — Reinforcement Learning (reasoning & alignment)
# Let the model try many solutions to each problem; reward the good ones. For subjective tasks, train a "Reward Model" to stand in for human judgement (**RLHF**). This is where "chains of thought" emerge. *Skipped in this notebook.*
#
# ---
#
# ### Learning Moment — map of this notebook to Stage 1
#
# | Karpathy step | Cell in this notebook | Real-world scale | Our toy scale |
# |---|---|---|---|
# | Data collection | Step 1 (dataset) | FineWeb ≈ **44 TB**, ~15T tokens | 32k names, ~230 KB |
# | Filtering (malware / PII / spam) | Step 1 — the `line.strip() … if line.strip()` filter | Hundreds of pipeline stages | One whitespace filter |
# | Tokenization | Step 2 (tokenizer) | GPT-4 BPE, **~100,000 tokens** | Char-level, **27 tokens** |
# | Neural network (transformer) | Steps 4 + 5 (params + `gpt`) | Hundreds of **billions** of params | **4,192** params |
# | Training objective (next-token prediction) | Step 6 (training loop) | Trillions of tokens over weeks on thousands of GPUs | 1,000 steps on your laptop |
# | Result = base model | Step 7 (inference) | "Internet document simulator" | "Name simulator" — literally the same thing, smaller |
#
# After Step 7, pause and look at the output: names like `kamon`, `jaire`, `vialan`. That is **exactly** what a base model does — continues in the style of its training distribution. It is not a chatbot. It has no persona. It cannot answer a question. That gap is what Stages 2 and 3 would fill.

# %%
"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
Everything here is the ALGORITHM. Any bigger GPT is just the same ideas made faster.
"""

# No numpy, no torch — on purpose. We will rebuild the relevant bits by hand, from math + lists.
import os       # os.path.exists — to check if the dataset file is already downloaded
import math     # math.log, math.exp — the calculus functions we will need inside autograd
import random   # random.seed, random.choices, random.gauss, random.shuffle — all sources of randomness

# Seed the random number generator so every run produces identical results.
# Without this, your weights would initialise differently each time and you would get
# slightly different generated names. Reproducibility beats creativity while learning.
random.seed(42)  # Let there be order among chaos.

# %% [markdown]
# ## Step 1 — the dataset
#
# **Mental model:** imagine a stack of 32,000 index cards, each with one baby name on it. We will read them one at a time, in random order, and let the model look for patterns.
#
# Why names? Because they are short, constrained, and the model can learn meaningful structure (bigrams, endings, vowel–consonant patterns) in minutes on a laptop — no GPU needed.
#
# Source: Karpathy's `makemore` dataset (US baby names).
#
# ---
#
# ### Learning Moment — this IS Stage 1, step 1 ("Data Collection + Filtering")
#
# Real LLMs start the same way: download something huge (Common Crawl, then FineWeb = ~44 TB of filtered text) and turn it into a shuffled list of documents. The downstream training loop does not care whether each document is a baby name or a Wikipedia article — it just wants next-token-prediction examples.
#
# Spot the two mini-filters below that mirror what a real pipeline does at industrial scale:
#
# - `if line.strip()` — drops empty lines. A real pipeline drops malware, spam, PII, NSFW, duplicates, broken Unicode, and low-quality pages.
# - `random.shuffle(docs)` — removes ordering bias. Real pipelines shuffle *and* re-weight by quality (higher-quality sources sampled more often).
#
# The ideas are identical; only the scale differs.

# %%
# If we have never run this before, download the dataset of real baby names.
# The file has one name per line: "emma", "olivia", "noah", ...
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# Read the file, strip whitespace, drop empty lines.
# `docs` is now: ["emma", "olivia", "noah", ...]  — a plain list of strings.
docs = [line.strip() for line in open('input.txt') if line.strip()]

# Shuffle so the model does not see names in alphabetical order (that would bias early learning).
random.shuffle(docs)

# Enriched summary so the numbers have context.
avg_len = sum(len(d) for d in docs) / len(docs)
total_chars = sum(len(d) for d in docs)
print(f"num docs: {len(docs)}")
print(f"  -> {len(docs):,} real baby names, one per line.")
print(f"  -> average length: {avg_len:.1f} chars  |  longest: {max(len(d) for d in docs)} chars")
print(f"  -> total characters: ~{total_chars/1024:.0f} KB")
print(f"  -> (real LLM pre-training uses ~44 TB, about 200 million x more data)")

# %% [markdown]
# ## Step 2 — the tokenizer
#
# **Mental model:** a **translator** between text and numbers, because neural networks only do arithmetic. It works in two directions, and both matter:
#
# - **Encode** (text → numbers): turn human-readable input into something the model can do math on. Used on the way *in*.
# - **Decode** (numbers → text): turn the model's numeric output back into human-readable text. Used on the way *out*.
#
# Both directions share a **single lookup table** — the list of every character the model knows:
#
# | direction | what we have | what we want | how to do it |
# |---|---|---|---|
# | **Encode** | a character, e.g. `'n'` | its integer ID, e.g. `13` | find the character's position in the list → `uchars.index('n')` |
# | **Decode** | an integer ID, e.g. `13` | the character it stands for, `'n'` | index into the list at that position → `uchars[13]` |
#
# Same table, used two ways. Tokenization is the **boundary** between the human world (strings) and the model world (integers) — every byte crossing that boundary, in either direction, passes through this translator.
#
# ---
#
# ### What is `uchars`? (the lookup table itself)
#
# `uchars` is just a Python **list of every unique character that appears in the dataset**, in sorted order. It *is* the lookup table referenced above. It gets built in one line:
#
# ```python
# uchars = sorted(set(''.join(docs)))
# ```
#
# That line does three things, right to left:
#
# | step | what happens | result |
# |---|---|---|
# | `''.join(docs)` | Glue every name in the dataset into one giant string | `"emma" + "olivia" + "noah" + ...` → `"emmaolivianoah..."` |
# | `set(...)` | Drop duplicates — keep only the *unique* characters | `{'e', 'm', 'a', 'o', 'l', 'i', 'v', 'n', 'h', ...}` |
# | `sorted(...)` | Arrange them in a stable, predictable order (alphabetical) | `['a', 'b', 'c', ..., 'y', 'z']` |
#
# For the names dataset this produces exactly the 26 lowercase letters `['a', 'b', ..., 'z']` — nothing else appears in the raw data.
#
# **Why `set` first?** Even though the dataset contains hundreds of thousands of characters total, only **26 are unique**. The model only needs to learn about each distinct character once — `set` collapses duplicates.
#
# **Why `sorted`?** Stability across runs. Python's `set` does not guarantee order, so without `sorted`, `'a'` might be token `5` one run and token `17` the next. Your trained model's weights would be scrambled. Sorting pins the assignment permanently.
#
# **The crucial trick — the index IS the token ID.** Because `uchars` is an ordered list, every character gets a natural integer identity: *its position in the list.*
#
# ```
# uchars[0]  == 'a'     →   'a' has token ID 0
# uchars[13] == 'n'     →   'n' has token ID 13
# uchars[25] == 'z'     →   'z' has token ID 25
# ```
#
# This is why **one list handles both directions**:
# - **Encode** — "what ID is `'n'`?" → ask the list for the position of `'n'` → `uchars.index('n')` → `13`.
# - **Decode** — "what character is ID `13`?" → ask the list for the item at position `13` → `uchars[13]` → `'n'`.
#
# No second data structure needed. The list's ordering *is* the vocabulary mapping.
#
# ---
#
# ### The specifics for our toy
#
# Assign every unique character an integer ID: `a=0, b=1, ..., z=25`. Then add one extra token — **BOS** (Beginning Of Sequence, id `26`) — that acts as a bookend so the model knows *"a name is starting"* and *"a name just ended"*.
#
# Example (encode direction): `"ann"` → `[BOS, a, n, n, BOS]` → `[26, 0, 13, 13, 26]`.
# Example (decode direction): token `0` → `'a'`. Token `13` → `'n'`. Token `26` → (BOS — not printed).
#
# **Why BOS at both ends?** Because the first letter of a name is itself a prediction ("given nothing so far, what is a likely start?"), and so is *"when should I stop?"*. BOS gives the model a concrete symbol for both.
#
# ---
#
# ### Learning Moment — character-level vs. Byte Pair Encoding (BPE)
#
# Our tokenizer is the simplest possible: **one token per character**, vocab of **27**. Real LLMs use **Byte Pair Encoding (BPE)** — GPT-4's vocab is about **100,000 tokens**, where common chunks like `" the"`, `"ing"`, or `"def"` become *single* tokens.
#
# Why bigger vocabularies? Two reasons:
#
# 1. **Shorter sequences → cheaper compute.** A 1,000-word article is ~5,000 chars but only ~750 BPE tokens. The transformer's cost scales quadratically with sequence length, so fewer tokens = much faster training.
# 2. **Better generalisation.** If `"running"` is one token, the model can learn it as a concept. If it is 7 character tokens, the model has to reconstruct the concept from substrings every time.
#
# ### Sharp-edge preview — the "strawberry problem"
#
# Because real LLMs see `"strawberry"` as maybe 2–3 BPE tokens, not 10 letters, they struggle to count letters inside it. Same reason they think `9.11 > 9.9` (tokenized as `9 . 11` vs `9 . 9`, where `11 > 9` as numbers). Our char-level model would *not* have this problem — but it also cannot take advantage of chunk-level shortcuts. Tokenization is always a trade-off.

# %%
# Get every unique character that appears anywhere in the dataset, sorted for stable IDs.
# For the names dataset this is just the 26 lowercase letters a..z.
uchars = sorted(set(''.join(docs)))

# Reserve the next unused integer (len(uchars) == 26) as the BOS special token.
# BOS does not correspond to any real character — it is a signal, not a letter.
BOS = len(uchars)

# Total vocabulary: every real character + the one special token = 27.
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")
print(f"  -> {len(uchars)} lowercase letters (a-z) + 1 special BOS token")
print(f"  -> every character in every name maps to one of these {vocab_size} integers")
print(f"  -> (GPT-4 vocab is ~100,000 BPE tokens — ours is ~3,700x smaller)")

# How encoding/decoding looks in practice:
#   encode "ann":   [uchars.index(ch) for ch in "ann"]  ->  [0, 13, 13]
#   decode token 0: uchars[0]                            ->  'a'

# %% [markdown]
# ### Peek inside — the tokenizer in action
#
# Optional demo. Skip it and the model still trains fine. Run it to *see* the translator you just built.

# %%
# ====== DEMO — peek inside the tokenizer ======

print("The lookup table (uchars):")
print(uchars)
print(f"\nLength: {len(uchars)}  — our 26 lowercase letters.")
print(f"BOS is token ID {BOS}  — not a real character, a signal.\n")

# Encode direction: text  ->  numbers
name = "ann"
encoded = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
print(f"Encode '{name}'  ->  {encoded}")

# Decode direction: numbers  ->  text
# Simulate a sampled sequence [BOS, k, a, t, e, BOS]
sampled_ids = [BOS, 10, 0, 19, 4, BOS]
decoded = [('<BOS>' if i == BOS else uchars[i]) for i in sampled_ids]
print(f"Decode {sampled_ids}  ->  {decoded}")
print(f"Visible name (BOS stripped): '{''.join(ch for ch in decoded if ch != '<BOS>')}'")

# %% [markdown]
# ## Interlude — derivatives, gradients, and why training is calculus
#
# The next cell (Step 3) builds an "autograd engine" — the machinery that lets a neural network learn. Before diving in, let's build the intuition for *why* any of this is needed. If you understand these six ideas, the code that follows becomes obvious.
#
# ---
#
# ### 1. Why do we even need derivatives?
#
# Training a model means solving this puzzle:
#
# > *"I have 4,192 knobs and one loss number. For each knob, tell me: should I turn it up or down, and by how much, to make the loss smaller?"*
#
# A **derivative** is the mathematical name for *exactly that answer*. It tells you how a tiny change in an input affects an output. No derivatives → no way to know which direction to turn any knob → no training.
#
# ---
#
# ### 2. What a derivative really is — "slope under your finger"
#
# Pick any smooth function and zoom in far enough. It stops looking curvy and starts looking like a **straight line**. The derivative at that point is just **the slope of that line**.
#
# - If slope is **+3**: nudging input up by `0.01` nudges output up by `0.03`. Output goes up 3× as fast as input.
# - If slope is **-2**: nudging input up by `0.01` nudges output *down* by `0.02`. Output falls.
# - If slope is **0**: input doesn't matter right here (you're at a flat spot — maybe a minimum, maybe a maximum).
#
# That is literally all a derivative is: *how sensitive is the output to the input, right now?* No calculus textbook required.
#
# **Examples you will see in the code:**
#
# | Operation | Derivative | Intuition |
# |---|---|---|
# | `c = a + b` | `dc/da = 1`, `dc/db = 1` | Nudge `a` up by `δ` → `c` goes up by exactly `δ`. Same for `b`. Addition passes change through 1-for-1. |
# | `c = a * b` | `dc/da = b`, `dc/db = a` | Nudge `a` up by `δ` → `c` goes up by `b*δ` (because `c` is `b` copies of `a`). Symmetric for `b`. |
# | `c = log(a)` | `dc/da = 1/a` | Big `a` → tiny effect from nudging (log grows slowly). Small `a` → huge effect. |
#
# When you look at the `Value` class code, `(1, 1)` for add and `(other.data, self.data)` for multiply are **literally these derivative rules** stored as Python tuples. Nothing more mysterious than that.
#
# ---
#
# ### 3. The chain rule — a relay race of slopes
#
# The above handles one operation in isolation. But a neural network is thousands of operations stacked. How do we get from *"the slope of `a * b`"* to *"the slope of `loss` w.r.t. some weight 15 operations deep in the graph"*?
#
# **The chain rule.** If `loss` depends on `x`, which depends on `y`, which depends on a weight `w`:
#
# ```
# d(loss)/d(w)  =  d(loss)/d(x)  *  d(x)/d(y)  *  d(y)/d(w)
# ```
#
# **Relay race analogy.** Each stage of the computation passes a "sensitivity baton" to the next. The local slope at each stage multiplies into the running total. At the end you have the end-to-end slope from loss all the way back to `w`, without ever having to write down the giant composed formula.
#
# **This is exactly what `backward()` does.** It walks the computation graph from loss back to every input, multiplying local slopes along the way.
#
# ---
#
# ### 4. Gradient = a vector of slopes, one per parameter
#
# A **gradient** sounds fancy but is simple: it is just *"the list of all 4,192 derivatives, one per parameter"*. Not one slope — a whole vector of them, one per knob.
#
# Geometrically, think of the loss as a **landscape** where every parameter is one axis (4,192-dimensional, but the 2D picture works fine for intuition). At any point in this landscape, the gradient points **uphill** — the direction of steepest increase. So `-gradient` points **downhill** — the direction that reduces loss fastest.
#
# **The blindfolded hiker analogy.** You're dropped onto a 4,192-D mountain. You can't see anything, but at your feet you can feel the slope in every direction (that's the gradient). Recipe:
#
# 1. Feel the downhill direction.
# 2. Take a small step that way.
# 3. Repeat.
#
# Eventually you reach a valley (low loss). *Training is literally this.* Adam (Step 6) is just a smarter version of this hiker with momentum and adaptive step sizes.
#
# ---
#
# ### 5. Why "auto"-grad? Because doing this by hand is impossible
#
# For a network with 4,192 parameters and thousands of operations per forward pass, computing every derivative by hand is a nightmare. But here's the trick:
#
# > **Every primitive operation (`+`, `*`, `log`, `exp`, `relu`) has a known local derivative rule. If we just record which operations happened in what order, a computer can apply the chain rule automatically.**
#
# Two pieces of machinery make this work:
#
# - **The computation graph.** As you do math, each new value remembers which values produced it and how. This builds a directed graph: inputs at the bottom, loss at the top, operations as edges. You do not construct this graph explicitly — it builds itself as a side effect of every `+` and `*`.
# - **Backpropagation.** Walk the graph backwards from loss. At each node, apply its local derivative rule and multiply into the running gradient (chain rule in action). Deposit the final gradient on each input.
#
# That's it. That is the entire trick behind every neural network library — PyTorch, TensorFlow, JAX — all the way up to what trains GPT-4. You are about to build the full thing in ~40 lines of Python.
#
# ---
#
# ### 6. Lock-in — the three pieces you're about to see in the Value class
#
# When you look at the code, every line maps to one of these:
#
# | What you'll see | What it does |
# |---|---|
# | `self.data` | The actual number (forward pass) |
# | `self._children` | Pointers back to the Values that produced this one — **the computation graph** |
# | `self._local_grads` | The local derivative rule for the op that produced this node (from section 2's table) |
# | Operator overloads (`__add__`, `__mul__`, etc.) | How the graph gets built automatically as you do math |
# | `self.grad` | Where the chain-rule gradient gets deposited |
# | `backward()` | The graph traversal: topological sort, then walk backwards multiplying slopes |
#
# **Three-sentence summary of the whole cell coming up:**
#
# > A `Value` object holds (a) a number, (b) pointers to the Values that produced it, and (c) the local derivative rule for the operation that produced it. Forward pass builds the graph automatically via Python's operator overloading. Backward pass walks the graph in reverse, applying the chain rule, depositing a gradient on every node.
#
# That is autograd. Every neural network library you will ever use is a scaled-up, hardware-accelerated version of what's in the next cell.

# %% [markdown]
# ## Step 3 — autograd: a number that remembers its own history
#
# **This is the most conceptually important cell in the notebook. Read slowly.**
#
# ### The problem
#
# Training a neural network means asking, for every one of the 4,192 internal weights:
#
# > *"If I nudged you up by a hair, would the final loss go up or down, and by how much?"*
#
# That is a **derivative**. With thousands of parameters and thousands of operations between them and the loss, we need an automatic way to compute all those derivatives.
#
# ### The trick — automatic differentiation
#
# Every number in the computation is wrapped in a `Value` object. A `Value` knows:
# - its numeric **data** (like a regular float)
# - its **grad** (filled in later, during backprop)
# - which other `Value`s it was built from (its **children**)
# - the **local derivative** with respect to each child
#
# Think of it like a **breadcrumb trail**: as we do math on `Value`s, each new `Value` drops a breadcrumb back to the ones that produced it. Later, starting at the loss and walking the trail backwards, we use the **chain rule** to fill in every gradient along the way.
#
# ### Two phases
#
# - **Forward:** do math normally; breadcrumbs accumulate automatically via Python's operator overloading (`+`, `*`, etc.).
# - **Backward:** start at the loss with `grad=1`, and recursively distribute gradient back to every contributor.
#
# ### Why it works (chain rule intuition)
#
# If `c = a * b`, then nudging `a` up by a tiny amount `δ` nudges `c` up by `b * δ` (because `c` is `b` times `a`). So the **local gradient of `c` with respect to `a`** is `b`. That is why you will see `(other.data, self.data)` as the local grads for multiplication in the code below.

# %%
class Value:
    """A scalar that remembers every operation it took part in, so we can backprop later."""

    # __slots__ tells Python: this class only ever has these four attributes.
    # Saves memory vs. the default __dict__ approach. Matters once you have millions of Values.
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # the actual numeric value (set during the forward pass)
        self.grad = 0                   # ∂loss/∂self — filled in during the backward pass
        self._children = children       # the Values this one was computed from
        self._local_grads = local_grads # ∂self/∂child for each child  (the chain-rule link)

    # Readable printing: when you `print(some_value)` you get the number + grad, not a memory address.
    # Purely a quality-of-life thing for the demo cells and for debugging.
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- Operator overloading ------------------------------------------------------------------
    # Python magic methods make `a + b` and `a * b` work on Values. Each returns a NEW Value
    # whose children are the inputs. The local grads encode the derivative of that operation.

    def __add__(self, other):
        # d(a+b)/da = 1,  d(a+b)/db = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # d(a*b)/da = b,  d(a*b)/db = a
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    # Power: d(a^n)/da = n * a^(n-1)
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    # Log: d(log a)/da = 1/a
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    # Exp: d(e^a)/da = e^a  (equal to the output itself)
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    # ReLU: pass through if positive, zero otherwise. Derivative is 1 where a>0 else 0.
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # Syntactic sugar: negation, reverse-ops (so `2 + value` works as well as `value + 2`),
    # subtraction, and division — all defined in terms of the primitives above.
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """Walk the computation graph backwards from self, filling in .grad on every Value."""

        # Build a topological ordering: every node appears AFTER all its children.
        # That way, when we traverse in reverse, every node is visited only after everything
        # that depends on it. This matters so gradients FULLY accumulate before being passed on.
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Seed: ∂self/∂self = 1. This is the starting point of the chain.
        self.grad = 1

        # Walk from output (loss) back to inputs, distributing gradient via the chain rule.
        # For each node v, for each child:
        #     child.grad  +=  (∂v/∂child) * (∂loss/∂v)
        # The `+=` is crucial — a Value used in multiple places collects gradient from all of them.
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# %% [markdown]
# ### Peek inside — autograd on a tiny problem
#
# The `Value` class is conceptually heavy. Let's prove it works on a problem small enough to verify by hand with high-school calculus.
#
# **Problem:** compute `c = a*b + a` with `a=2, b=3`, then ask: what are `dc/da` and `dc/db`?
#
# **By hand:**
# - `c = 2*3 + 2 = 8` ✓
# - `dc/da = b + 1 = 4`   (product rule: `d(a*b)/da = b`; the standalone `+a` contributes `+1`)
# - `dc/db = a = 2`       (product rule: `d(a*b)/db = a`; `b` does not appear elsewhere)
#
# If autograd is working, calling `c.backward()` should populate exactly those numbers in `a.grad` and `b.grad`. Let's check.

# %%
# ====== DEMO — autograd on a problem we can verify by hand ======

a = Value(2.0)
b = Value(3.0)
c = a*b + a

print(f"Forward:  c = a*b + a")
print(f"          c.data = {c.data}   (expected 8)\n")

c.backward()

print(f"After c.backward():")
print(f"   a.grad = {a.grad}   (expected 4  =  b + 1)")
print(f"   b.grad = {b.grad}   (expected 2  =  a)")

print("\nThese gradients match hand-calculus exactly.")
print("loss.backward() below will do the SAME thing, just over a graph with thousands of nodes.")

# %% [markdown]
# ## Interlude — what *is* a neural network, really?
#
# Before we look at parameters, let's build the mental model from the ground up. If you walk away with just this section, you understand 80% of how every neural network (and every transformer) works.
#
# ---
#
# ### 1. The neuron — a weighted sum followed by a squish
#
# The smallest unit. A neuron takes in a list of numbers, multiplies each by a **weight**, adds them up, and passes the result through a **non-linear "squish"**. In pseudocode:
#
# ```
# output = squish(w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + bias)
# ```
#
# **Analogy.** Think of a neuron as a little judge on a panel. Each input is a piece of evidence. The **weights are how much this judge cares about each piece of evidence** (high weight = "I heavily weight this input"; near-zero weight = "I don't care"; negative = "this argues against"). The judge adds up the weighted evidence and fires if the total is convincing enough.
#
# **The weights ARE the knowledge.** A neuron with different weights answers the same question differently. Training = slowly adjusting the weights until the panel of judges collectively reaches correct verdicts.
#
# ---
#
# #### What is a "squish" exactly?
#
# *"Squish"* is informal slang for a **non-linear activation function** — a simple function with a bend, curve, or threshold that you apply to the weighted sum *after* computing it. The term comes from the way these functions take any real number (positive, negative, huge, tiny) and *squeeze/squish* it through a non-linear shape before it moves to the next layer.
#
# The squish used in this notebook is **ReLU** (Rectified Linear Unit), and it's the simplest one possible:
#
# ```python
# def relu(x):
#     return max(0, x)     # if x > 0, keep it; otherwise return 0
# ```
#
# That is the *entire* ReLU non-linearity. One `max()` call. If the weighted sum is positive, the neuron fires proportionally; if it's negative, the neuron stays silent. That's it.
#
# Other common squishes you'll encounter elsewhere — all sharing the property that they **have a bend or curve** (they are NOT straight lines):
#
# | squish | what it does | where you see it |
# |---|---|---|
# | **ReLU** | `max(0, x)` — flat zero for negatives, linear for positives | modern deep learning default, used in this notebook |
# | **sigmoid** | smoothly maps any input to (0, 1) | old-school; classification output layers |
# | **tanh** | smoothly maps any input to (-1, +1) | old recurrent networks |
# | **GeLU** | smoother ReLU-like curve | GPT-2, GPT-3, real LLMs |
#
# #### Why do we need a squish at all?
#
# This is the critical insight: **without the squish, stacking neural-network layers gains you nothing.**
#
# Suppose we had two layers with NO non-linearity between them:
#
# - Layer 1:  `y = W1 · x`
# - Layer 2:  `z = W2 · y` = `W2 · (W1 · x)` = `(W2 · W1) · x`
#
# The two matrix multiplies collapse into a single matrix `(W2 · W1)`. Two linear layers stacked is **mathematically equivalent to one linear layer**. No new expressive power. You could stack a million of them and still only represent straight-line relationships.
#
# **The squish is what breaks this collapse.** Insert a ReLU between the layers and suddenly `z = W2 · relu(W1 · x)` is genuinely more expressive than any single linear layer. The bend in ReLU lets the network represent *piecewise* functions — different linear behaviour in different regions of input space. Stack many squish-separated layers and you can approximate almost any function at all (this is the "universal approximation" theorem).
#
# **Mental model for the squish:** think of it as a **decision point**. The weighted sum is the neuron's evidence; the squish is the threshold at which the neuron "fires". For ReLU the threshold is zero — below zero, stay silent; above zero, fire proportionally to the evidence. That threshold is where the non-linearity lives, and non-linearity is where learning becomes possible.
#
# > Without squishes → the model is a very fancy spreadsheet formula.
# > With squishes    → the model is something that can learn.
#
# You will see `xi.relu()` called exactly once inside the MLP in Step 5. That single line is what turns the whole transformer from a glorified matrix multiply into a learning machine.
#
# ---
#
# ### 2. A layer — many neurons side by side
#
# Stack, say, 64 neurons in parallel. Each one looks at the same inputs but has its **own set of weights**, so each one learns a different feature detector. One neuron might end up specialising in "is this a vowel?", another in "is this near the end of a word?" — nobody programs those specialisations; they *emerge* during training as useful things to notice.
#
# A layer is mathematically just a **matrix multiply**: inputs (a vector) times a weight matrix (one row per neuron) = outputs (a vector).
#
# This is exactly what our `linear(x, w)` function does. Read it again now:
#
# ```python
# def linear(x, w):
#     return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
# ```
#
# `w` is the weight matrix. One row per neuron. The inner `sum(...)` is one neuron's weighted sum. The outer list is every neuron in the layer firing at once. **That is a layer.**
#
# ---
#
# ### 3. A network — layers stacked back-to-back
#
# Feed the output of one layer into the next. Early layers learn simple features; later layers combine those into more abstract concepts. Between layers, apply a non-linearity (ReLU). Without the non-linearity, stacking layers gains you nothing — it would collapse mathematically into a single layer. *The non-linearity is what lets depth add power.*
#
# ---
#
# ### 4. "Weights" vs. "parameters" — same thing, different framing
#
# - **Weights** = the numbers inside the weight matrices that get multiplied by the inputs.
# - **Parameters** = every learnable number in the model, collectively. Weights are the dominant kind of parameter. (Real networks also have "biases"; this code skips them for simplicity.)
#
# When people say *"GPT-4 has 1.8 trillion parameters"* they literally mean 1.8 trillion individual floating-point numbers, each initialised randomly, each nudged a little every training step, each ending up encoding some tiny sliver of learned structure.
#
# **Where does "knowledge" live?** Nowhere specific. It is **distributed** — smeared across millions of weights, none of which individually mean anything. This is why you cannot open GPT-4 and point to "the fact about Paris being in France". It is a diffuse statistical residue of having seen the string `"Paris, France"` in the training data billions of times.
#
# *(Pause here. If you're new to neural networks, reread sections 1–4 once more before going on — the rest of the notebook treats these as established.)*

# %% [markdown]
# ## Interlude continued — how learning happens and how a transformer is shaped
#
# ### 5. How the network actually *learns*
#
# Three steps, repeated millions of times:
#
# 1. **Forward pass** — push data through the network to get a prediction.
# 2. **Compute loss** — measure how wrong the prediction was.
# 3. **Backward pass + update** — figure out which weights contributed to the error and nudge them down; the others up. This is where `Value.backward()` earns its keep.
#
# Every weight receives a tiny nudge every step. Over a million steps, random weights become meaningful weights. That is the entire miracle.
#
# ---
#
# ### 6. A transformer is a *particular recipe* of layers
#
# "Transformer" is not a new kind of math — it is a specific pattern of arranging linear layers, with two key innovations:
#
# - **Self-attention layer.** Each position in the sequence looks at every other position and decides what to pay attention to. Without attention, a model processes each token in isolation; with attention, tokens can *communicate*. This is the main reason transformers beat everything that came before.
# - **Feed-forward (MLP) layer.** A standard two-layer neural net (expand → ReLU → squeeze) applied independently to each position. This is where "per-position thinking" happens.
#
# Transformers alternate: attention → MLP → attention → MLP → ... for however many layers. That is it. That is the entire "secret".
#
# ---
#
# ### 7. What this notebook's network looks like in those terms
#
# Our `gpt` function is a **1-layer transformer** with:
#
# | Component | What it is, in "neurons and weights" terms |
# |---|---|
# | **`wte` (token embedding)** | A lookup table — not really "neurons", but still 27×16 = **432 learnable weights** that give each of the 27 tokens a 16-dim personality vector |
# | **`wpe` (position embedding)** | Same idea but for positions: 16×16 = **256 weights** |
# | **`attn_wq / wk / wv`** | Three layers of **16 neurons each**, each neuron looking at 16 inputs → 16×16 = 256 weights each. They produce Query / Key / Value |
# | **`attn_wo`** | Another 16-neuron layer (16×16 = 256 weights) that projects the concatenated head outputs back to model width |
# | **`mlp_fc1`** | A layer of **64 neurons** (wider than the rest) — 64×16 = **1,024 weights**. This is the "expand" step |
# | **`mlp_fc2`** | A layer of **16 neurons** each looking at 64 inputs — 16×64 = **1,024 weights**. The "squeeze" back to model width |
# | **`lm_head`** | The final layer: **27 neurons** each looking at 16 inputs → 27×16 = **432 weights**. Output = 27 scores, one per possible next token |
#
# Add them up: **4,192 parameters**. That matches the `num params: 4192` that will print when you run the next cell. Every single one of those 4,192 numbers is a weight on one of the "judges" in the network. Training is the process of finding a good setting for all of them simultaneously.
#
# **Mental model lock-in.** If any of this is still fuzzy, re-read sections 1–3 once more. After that, the actual code in the next cell is just bookkeeping: a Python `dict` that groups the weight matrices by name, and a flat `params` list that hands every weight to the optimizer. The *concepts* are all above.

# %% [markdown]
# ## Step 4 — the parameters (the "brain" of the model)
#
# Every number the model *learns* lives in `state_dict`. We start them random; training sculpts them. This step has two goals: (1) introduce every matrix the model owns, and (2) **derive the total parameter count from first principles**, so `num params: 4192` stops being a magic number and becomes an arithmetic consequence of the architectural choices.
#
# ---
#
# ### Hyperparameters — chosen by us, not learned
#
# These 5 numbers define the entire architecture. Every other size in the model is derived from them.
#
# | name | value | what it controls | why this value |
# |------|-------|------------------|----------------|
# | `n_layer` | 1 | how deep the network is (stack of transformer blocks) | 1 is enough to learn name-level structure; more layers = more capacity but more compute |
# | `n_embd` | 16 | the "width" — how many numbers represent each token internally | 16 is small but workable; real LLMs use 768–18,000 |
# | `block_size` | 16 | max name length we can process (longest real name is 15 chars) | tight fit for the dataset; would be 128k for GPT-4 |
# | `n_head` | 4 | number of parallel attention heads | gives the model 4 "perspectives" to attend from |
# | `head_dim` | 4 | `n_embd / n_head` = `16 / 4` — the size of each head's subspace | derived, not chosen; required for multi-head attention to split cleanly |
#
# **Important:** `n_head` must divide `n_embd` evenly. If you change one, you may have to change the other.
#
# ---
#
# ### The rule that determines every matrix shape
#
# A layer that maps `nin` inputs to `nout` outputs has a weight matrix of shape `(nout, nin)` — one row per output neuron, each row holding `nin` weights. Every matrix below follows this rule; the only question at each step is *"what are `nin` and `nout` right here?"*
#
# Walk through the forward pass once, top to bottom, and each shape becomes forced:
#
# | matrix | shape | how the shape is forced | parameters |
# |---|---|---|---|
# | `wte` (token embedding) | `(vocab_size, n_embd)` = `(27, 16)` | one 16-dim row per token ID; needs `vocab_size` rows to cover every token | 27 × 16 = **432** |
# | `wpe` (position embedding) | `(block_size, n_embd)` = `(16, 16)` | one 16-dim row per position; needs `block_size` rows to cover every position | 16 × 16 = **256** |
# | `attn_wq` (Query projection) | `(n_embd, n_embd)` = `(16, 16)` | projects the 16-dim token state into a 16-dim Query | 16 × 16 = **256** |
# | `attn_wk` (Key projection) | `(n_embd, n_embd)` = `(16, 16)` | same idea for Key | 16 × 16 = **256** |
# | `attn_wv` (Value projection) | `(n_embd, n_embd)` = `(16, 16)` | same idea for Value | 16 × 16 = **256** |
# | `attn_wo` (Output projection) | `(n_embd, n_embd)` = `(16, 16)` | merges the concatenated head outputs (4 heads × 4 dims = 16) back into a 16-dim vector | 16 × 16 = **256** |
# | `mlp_fc1` (MLP expand) | `(4*n_embd, n_embd)` = `(64, 16)` | standard transformer convention: expand to 4× width before the non-linearity | 64 × 16 = **1,024** |
# | `mlp_fc2` (MLP squeeze) | `(n_embd, 4*n_embd)` = `(16, 64)` | squeeze back to model width so residuals can stack | 16 × 64 = **1,024** |
# | `lm_head` (output projection) | `(vocab_size, n_embd)` = `(27, 16)` | turns the 16-dim "thought" into 27 scores, one per possible next token | 27 × 16 = **432** |
#
# ---
#
# ### Where the 4,192 comes from — the full sum
#
# Group the matrices by their role in the architecture:
#
# | bucket | contents | arithmetic | count |
# |---|---|---|---|
# | **Embeddings** | `wte` + `wpe` | `432 + 256` | **688** |
# | **Attention (per layer)** | `wq + wk + wv + wo` | `256 × 4` | **1,024** |
# | **MLP (per layer)** | `fc1 + fc2` | `1,024 + 1,024` | **2,048** |
# | **Output head** | `lm_head` | `432` | **432** |
#
# `1,024 + 2,048 = 3,072` per transformer layer. We have `n_layer = 1`, so:
#
# ```
# total = 688  (embeddings)
#       + 3,072 * 1  (transformer layers)
#       + 432  (output head)
#       = 4,192
# ```
#
# If you bumped `n_layer` from 1 to 2, the number would be `688 + 3,072 * 2 + 432 = 7,232`. The transformer layer is where scale compounds — everything else is fixed cost.
#
# ---
#
# ### Weight matrices — why each one exists (analogies)
#
# Now that you know the shapes, here is what each matrix is *for*:
#
# - **`wte` (token embedding):** a lookup table with one row per token. Row `5` is *"the personality of the letter `f`"* as 16 numbers. Initially random; training makes similar-behaving letters drift toward similar rows.
# - **`wpe` (positional embedding):** same shape as `wte`, but indexed by *position*, not letter. Row 3 is *"how being the 3rd letter in a name feels"*. This is how the model learns that names tend to start certain ways and end other ways.
# - **`attn_wq / wk / wv / wo`:** Query / Key / Value / Output projections for the attention mechanism (fully explained in Step 5). All `16 × 16` because attention doesn't change the model's width — it just rearranges information across positions.
# - **`mlp_fc1 / fc2`:** a tiny 2-layer network that does "per-token thinking" after each attention step. The `16 → 64 → 16` sandwich is a standard pattern: widen to create room for non-linear feature interactions, then narrow back so the output can be added to the residual stream.
# - **`lm_head`:** the final translator — takes the model's internal 16-number "thought" and maps it to 27 scores, one per possible next token. Interestingly it's the same shape as `wte` transposed; some models tie these weights together to save parameters, but we keep them separate for clarity.
#
# ---
#
# ### Scale check — how small is 4,192, really?
#
# | model | parameters | ratio to microGPT |
# |---|---:|---:|
# | **microGPT (this notebook)** | **4,192** | 1× |
# | GPT-2 small | 124 million | ~30,000× |
# | GPT-3 | 175 billion | ~42 million× |
# | GPT-4 (est.) | 1.8 trillion | ~430 million× |
#
# Doubling `n_embd` roughly **quadruples** the parameter count (because most matrices are `n_embd × n_embd`). Doubling `n_layer` roughly **doubles** the per-layer contribution. This is why scaling models up is so expensive — a "small" architectural bump can add billions of parameters.
#
# `params` is just a flat list of every single trainable `Value` — 4,192 of them. The optimizer iterates this list, updating each weight independently once per training step.

# %%
# --- Hyperparameters: the ARCHITECTURE knobs. We pick these; training does not change them. ---
n_layer = 1                  # one transformer block (deeper = stronger, but slower)
n_embd = 16                  # internal "width": every token becomes a vector of 16 numbers
block_size = 16              # max context length (longest real name is 15 chars; +1 for the BOS)
n_head = 4                   # number of parallel attention heads
head_dim = n_embd // n_head  # each head works in a 4-dim subspace (16 / 4)

# Helper: build a (nout x nin) matrix of Values, each initialised from a tiny Gaussian.
# std=0.08 keeps initial weights small so early gradients neither explode nor vanish.
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# state_dict holds every learnable matrix, keyed by name (a convention borrowed from PyTorch).
state_dict = {
    'wte':     matrix(vocab_size, n_embd),   # token embedding table: 27 rows x 16 cols
    'wpe':     matrix(block_size, n_embd),   # position embedding table: 16 rows x 16 cols
    'lm_head': matrix(vocab_size, n_embd),   # final projection: 16 inputs  ->  27 scores
}

# Per-layer weights. Only one layer here, but the loop makes it trivial to add more.
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)       # Query projection
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)       # Key projection
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)       # Value projection
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)       # Output projection (after heads merge)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)   # MLP expand:  16 -> 64
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)   # MLP squeeze: 64 -> 16

# Flatten every matrix into a single flat list of Values. The optimizer iterates this list.
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
print(f"  -> {len(params):,} learnable numbers. This IS the entire 'brain' of the model.")
print(f"  -> embeddings: {vocab_size*n_embd + block_size*n_embd:,}  |  "
      f"attention: {4*n_embd*n_embd*n_layer:,}  |  "
      f"MLP: {2*4*n_embd*n_embd*n_layer:,}  |  "
      f"output head: {vocab_size*n_embd:,}")
print(f"  -> real-world scale: GPT-2 small is ~30,000x larger, GPT-4 is ~430 million x larger")

# %% [markdown]
# ### Peek inside — what do the 4,192 parameters actually look like?
#
# The markdown above *claims* we have matrices of specific shapes summing to 4,192 parameters, all Gaussian-initialised with std 0.08. This cell **proves** it. You'll see:
#
# - the shape of every `state_dict` entry, with parameter counts per matrix (should add up to 4,192),
# - a few actual `Value` objects from `wte[0]` — these are the 16 numbers that currently define the "personality" of the letter `a`,
# - summary statistics of the raw initialisation: min, max, mean, standard deviation.

# %%
# ====== DEMO — parameter shapes and initial values ======

print("state_dict — shapes and parameter counts:\n")
total = 0
for name, mat in state_dict.items():
    rows, cols = len(mat), len(mat[0])
    count = rows * cols
    total += count
    print(f"  {name:25s}  shape=({rows:2} x {cols:2})  =  {count:5} params")
print(f"  {'TOTAL':25s}                       =  {total:5} params")

# Peek at the actual numbers that define token 'a' right now
print(f"\nFirst 5 of the 16 numbers in wte[0]  (the 'personality' of letter 'a', pre-training):")
for v in state_dict['wte'][0][:5]:
    print(f"   {v}")
print("   ... (11 more)")

# Sanity-check the random initialisation.
# Why this matters: good init is a pre-condition for training at all.
#   - mean far from 0  -> every neuron biased in the same direction; symmetry of learning broken
#   - std too large    -> activations explode across layers, forward pass becomes NaN
#   - std too small    -> signal vanishes layer-by-layer, gradients can't flow, model never learns
# We initialised with random.gauss(0, 0.08), so we expect mean ~0 and std ~0.08.
all_vals = [p.data for p in params]
mean = sum(all_vals) / len(all_vals)
std  = (sum((x - mean) ** 2 for x in all_vals) / len(all_vals)) ** 0.5

print(f"\nRaw statistics of all {len(params)} parameters:")
print(f"   min  = {min(all_vals):+.4f}   (~3 standard deviations below zero — a rare tail event)")
print(f"   max  = {max(all_vals):+.4f}   (~3 standard deviations above zero — the other tail)")
print(f"   mean = {mean:+.4f}   target: 0.0     — any bias would tilt every neuron in one direction")
print(f"   std  = {std:.4f}    target: 0.08    — our chosen initialisation scale")

# Render a compact histogram so the bell-curve shape is visible, not just asserted.
print(f"\nDistribution shape (should look like a bell curve centered on 0):")
edges = [-0.25 + 0.05 * i for i in range(11)]          # 10 bins from -0.25 to +0.25
bins = [0] * 10
for val in all_vals:
    for b in range(10):
        if edges[b] <= val < edges[b + 1]:
            bins[b] += 1
            break
peak = max(bins)
for b, count in enumerate(bins):
    bar = '#' * int(40 * count / peak)
    print(f"   [{edges[b]:+.2f}, {edges[b+1]:+.2f})  {bar}  ({count})")

print(f"\nThe bell shape is not cosmetic — it is what keeps training numerically stable.")
print(f"Later, after training, repeat this histogram and watch it change: a few weights will drift")
print(f"into longer tails as the model discovers structure, but the bulk will still be near zero.")

# %% [markdown]
# ## Step 5 — the model: a transformer in one function
#
# The `gpt` function IS the entire model. Given one token at one position (plus the history of previous tokens' keys/values), it returns 27 scores — one per possible next token.
#
# ### The pipeline for one token
#
# ```
# [token_id, pos_id]
#     |
#     v
#  wte[token] + wpe[pos]         "what letter am I? where am I?"  ->  16-number vector
#     |
#     v
#  rmsnorm                        keep the vector's magnitude sane
#     |
#     v
#  Attention block                look at the past, decide what's relevant now
#     |
#     v
#  MLP block                      think about it some more
#     |
#     v
#  lm_head                        project to 27 scores (logits)
# ```
#
# ### Attention — the study-group analogy
#
# A classroom of students (one per position so far) sits around a table. The current student wants to decide how to update their notes. So:
#
# 1. They write a **Query** — *"what am I looking for right now?"* (derived from their current state).
# 2. Every student (including themselves) holds up a **Key** — *"here is a label advertising what I know"*.
# 3. Every student also has some **Value** content — *"here is what I would actually tell you if you listened"*.
#
# The current student compares their Query to everyone's Key, gets a similarity score for each, softmaxes those into a probability distribution, and takes a weighted sum of Values. Done — they have "attended" to the past.
#
# **Multi-head:** instead of one study group, run 4 in parallel. Each head's Q/K/V operates on a 4-dim slice. After training, different heads specialise in different kinds of relationships (vowel proximity, position within the name, etc.). Their outputs are concatenated.
#
# ### Causal masking — why the past-only rule matters
#
# The model must predict *the next token given only tokens it has seen so far*. If it could peek at the future, training would collapse into a trivial identity problem: "what letter comes next?" would be answered by "look at the next letter" — learning nothing.
#
# In real transformer code, causality is enforced with a *mask* that zeroes out attention scores for future positions. **In our implementation it is implicit and free.** We only ever append to `keys[li]` and `values[li]` up to the *current* position before calling attention, so there are literally no future keys/values to attend to. Same effect, no mask needed.
#
# ### The KV cache — a speed hack, not architecture
#
# Notice that `gpt()` takes `keys` and `values` as arguments and *appends to them*. These lists are a **cache**, not part of the architecture. Here's why they exist:
#
# When generating token 10, the model needs Q for token 10 (new), but it also needs K and V for tokens 0–9 (already computed during previous calls). Re-running the full forward pass for every past token every time would be wasteful. Instead, each call:
#
# - computes `q`, `k`, `v` for *the current token only*,
# - **appends** `k` and `v` to the cache,
# - computes attention using the current `q` against the full cached history of `k`s and `v`s.
#
# Q is not cached — it's only needed for the current position. K and V *are* cached because every future position will reuse them. This is the **KV cache**, and at real scale it's what makes LLM inference fast. For this toy it just makes the code cleaner; at GPT-4 scale it is the difference between generating 10 tokens/second and 1000.
#
# ### MLP — "think for yourself"
#
# After attention mixes info across positions, the MLP does a per-token non-linear transformation: expand to 4× width, apply ReLU, squeeze back. Intuition: **attention gathers information; MLP processes it.**
#
# ### Residual connections (the `a + b` at the end of each block)
#
# Each block *adds* its output to its input instead of replacing it. Analogy: a student's notes get annotated, not rewritten. This dramatically eases training — gradients flow through the `+` unchanged, which lets signals reach early layers even in very deep networks.
#
# ### RMSNorm — volume control for activations
#
# A normalisation step that rescales vectors to roughly unit magnitude. Pseudocode:
#
# ```
# rmsnorm(x) = x / sqrt(mean(x^2) + tiny_epsilon)
# ```
#
# **Why this matters.** Recall the parameter-init demo (Step 4): we chose `std=0.08` precisely because initialisations that are "too big" cause activations to explode layer-by-layer (NaN), and "too small" cause them to vanish (no signal). RMSNorm enforces the same discipline *during the forward pass*: after every block, rescale activations to keep them in a healthy range no matter what the weights are doing. It's the reason you can stack 100 transformer layers without the signal collapsing or exploding.
#
# **RMSNorm vs. LayerNorm:** the original GPT-2 used LayerNorm, which also subtracts the mean. RMSNorm skips the subtraction and is slightly cheaper — modern LLMs (LLaMA, this toy) use it.
#
# ### The softmax numerical trick
#
# You'll see `softmax` defined with a surprising-looking line:
#
# ```python
# max_val = max(val.data for val in logits)
# exps = [(val - max_val).exp() for val in logits]
# ```
#
# Why subtract the max? Because `exp()` grows *very* fast — `exp(800)` overflows to infinity in double-precision, and softmax then produces `NaN`. Subtracting the max shifts every logit downward so the biggest exponent is 0 (and thus `exp(0)=1`, the largest possible non-overflow). **The output distribution is mathematically unchanged** because softmax is invariant to adding a constant to all its inputs — ratios of exponentials stay the same. Standard trick in every real ML library.
#
# ---
#
# ### Learning Moment — same architecture as GPT-4, different scale
#
# The `gpt` function below is a **complete transformer**. Scale it up and you get every modern LLM:
#
# | Knob | microGPT (this notebook) | GPT-2 small | GPT-3 | GPT-4 (estimated) |
# |---|---|---|---|---|
# | `n_layer` | **1** | 12 | 96 | ~120 |
# | `n_embd` | **16** | 768 | 12,288 | ~18,000 |
# | `n_head` | **4** | 12 | 96 | ~128 |
# | `block_size` (context) | **16** | 1,024 | 2,048 | 128,000 |
# | total params | **~4 K** | 124 M | 175 B | ~1.8 T |
#
# Same function. Same Q/K/V attention. Same residuals. Same RMSNorm. Just bigger numbers and much more data. Once you understand this cell, you understand the *architecture* of every frontier LLM.
#
# ### Sharp-edge preview — context window == working memory
#
# `block_size = 16` is literally this model's **working memory**. It can only look at 16 tokens of history; anything earlier is gone. For GPT-4 it is 128,000 tokens — enormous, but still finite.
#
# Karpathy's point: parameters = **vague recollection** (baked in during training, fuzzy and compressed). Context window = **working memory** (sharp, exact, but ephemeral). That is why you should *paste the relevant document into the chat* rather than rely on the model to "remember" it — pasting moves knowledge from vague recollection into working memory.

# %% [markdown]
# ### Peek inside — attention, on actual numbers (before you read the code)
#
# The study-group analogy is fine as a mental model, but you should *see* attention work on concrete numbers before you encounter it buried inside the `gpt` function. This demo does exactly that, using only Python's built-in `math` — no `Value` objects, no helpers. Just arithmetic.
#
# **Setup:** imagine the model is partway through processing the name `"em"` and is about to process the next position. The context so far is 3 tokens: `[BOS, 'e', 'm']`. Each token has a 4-dimensional hidden state (in the real model it's 16-dim, but 4 fits on screen). We'll pretend projection matrices have already been applied, so the K and V vectors below are just what the model "sees" when it looks at each past token.
#
# **What you'll watch:**
#
# 1. The current token's Query vector (its question).
# 2. The three past tokens' Key vectors (their labels).
# 3. Raw similarity scores = `Q · K` for each past token.
# 4. Scaled + softmaxed → attention weights (must sum to 1).
# 5. Output = weighted sum of the Value vectors.
#
# This is the **entire** attention mechanism. Everything in `gpt()` below is just scaling this up to 16 dimensions, 4 parallel heads, and wrapping the weights in `Value` objects so autograd can track them.

# %%
# ====== DEMO — attention from first principles, pure Python ======
import math as _m  # local alias so we don't shadow anything

# Three past tokens: BOS, 'e', 'm'. Four-dim vectors (handcrafted for illustration).
past_labels = ['BOS', "'e'", "'m'"]
K = [
    [ 0.1,  0.9, -0.2,  0.0],   # key for BOS  — "I am a start marker"
    [ 0.8,  0.1,  0.3, -0.2],   # key for 'e'  — "I am a vowel"
    [ 0.2, -0.1,  0.7,  0.5],   # key for 'm'  — "I am a consonant, mid-word"
]
V = [
    [1.0, 0.0, 0.0, 0.0],       # value for BOS
    [0.0, 1.0, 0.0, 0.0],       # value for 'e'
    [0.0, 0.0, 1.0, 0.0],       # value for 'm'
]

# Current token's Query — the 'question' it is asking. Pretend this is what
# 'm' produced after its wq projection: "I'm looking for another consonant-y thing."
Q = [0.3, -0.1, 0.7, 0.3]

# --- Step 1: raw similarity between Q and each past K (dot product) ---
scores = [sum(Q[i] * K[t][i] for i in range(4)) for t in range(3)]
print("Step 1  raw similarities  Q . K:")
for label, s in zip(past_labels, scores):
    print(f"   vs. {label:4}  ->  {s:+.3f}")

# --- Step 2: scale by sqrt(head_dim). Standard transformer trick. ---
head_dim = 4
scaled = [s / _m.sqrt(head_dim) for s in scores]
print(f"\nStep 2  scaled (divided by sqrt({head_dim}) = {_m.sqrt(head_dim):.3f}):")
for label, s in zip(past_labels, scaled):
    print(f"   vs. {label:4}  ->  {s:+.3f}")

# --- Step 3: softmax to get probabilities that sum to 1 ---
max_s = max(scaled)                           # numerical-stability max trick
exps = [_m.exp(s - max_s) for s in scaled]
total = sum(exps)
weights = [e / total for e in exps]
print("\nStep 3  softmax -> attention weights (must sum to 1):")
for label, w in zip(past_labels, weights):
    bar = '#' * int(w * 40)
    print(f"   on {label:4}  w={w:.3f}  {bar}")
print(f"   sum = {sum(weights):.4f}")

# --- Step 4: weighted sum of value vectors = the head's output ---
output = [sum(weights[t] * V[t][i] for t in range(3)) for i in range(4)]
print(f"\nStep 4  output = sum_t ( weight_t * V[t] ):")
print(f"   = {[round(x, 4) for x in output]}")

print("\nThat is attention. The current token has 'read' from its past by weighting each")
print("past Value by how relevant (similarity-wise) the past Key was to its Query.")

# %%
# --- Three primitive operations we need inside gpt() ---

def linear(x, w):
    """Matrix-vector multiply: for each row of `w`, take the dot product with `x`.
    If x has length nin and w is (nout x nin), the result has length nout.
    Geometrically, this is a linear transformation — rotate, scale, and reflect the input."""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """Turn an arbitrary list of numbers into a probability distribution (sums to 1).
    Numerical-stability trick: subtract the max before exponentiating, to prevent exp(big) = inf.
    The subtraction cancels out mathematically but keeps the arithmetic well-behaved."""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """Root-Mean-Square normalisation: rescale `x` so its typical magnitude is ~1.
    Pseudocode: scale = 1 / sqrt(mean(x^2) + tiny_epsilon), then x * scale.
    Without this, values can grow or shrink layer-by-layer and training diverges."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5   # +1e-5 avoids division by zero if every x is zero
    return [xi * scale for xi in x]


# --- The model: one token in, 27 scores (logits) out ---

def gpt(token_id, pos_id, keys, values):
    """Run one forward step of the model for a single token at a single position.

    `keys` and `values` are LISTS that we APPEND to as we process each position — this builds
    up the memory of what the past looked like, so future tokens can attend to it (KV cache)."""

    # 1. Embed. Look up the token's row and the position's row, then add them.
    #    Intuition: "I am letter 'a' AND I am the 3rd letter of the name" -> one vector.
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)   # normalise once before the first block

    for li in range(n_layer):
        # ============== 1) Multi-Head Self-Attention ==============
        x_residual = x           # remember the input so we can add it back (residual connection)
        x = rmsnorm(x)

        # Project the input into THREE different 16-dim spaces.
        q = linear(x, state_dict[f'layer{li}.attn_wq'])   # my question
        k = linear(x, state_dict[f'layer{li}.attn_wk'])   # my label  (what I know)
        v = linear(x, state_dict[f'layer{li}.attn_wv'])   # my content (what I would share)

        # Append my K and V to the running KV cache. Future tokens will read this history.
        keys[li].append(k)
        values[li].append(v)

        # Process each head independently, then concatenate their outputs.
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim                                            # this head's slice start
            q_h = q[hs:hs+head_dim]                                      # my Q for this head
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]                # everyone's K so far
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]              # everyone's V so far

            # Similarity between my Q and each past K. Scaled by sqrt(head_dim) to keep the
            # variance of the dot products stable regardless of head size — a standard trick
            # from the original "Attention Is All You Need" paper.
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                           for t in range(len(k_h))]

            # Softmax those similarities into weights that sum to 1.
            attn_weights = softmax(attn_logits)

            # Weighted sum of past Values = this head's output.
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                        for j in range(head_dim)]
            x_attn.extend(head_out)   # concatenate heads: 4 heads * 4 dims  ->  back to 16 dims

        # Final linear projection after the heads are concatenated back together.
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]   # residual: attention ADDS info, does not replace

        # ============== 2) MLP block ("think for myself") ==============
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # expand:  16 -> 64
        x = [xi.relu() for xi in x]                        # non-linearity — this is what makes an MLP > linear
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # squeeze: 64 -> 16
        x = [a + b for a, b in zip(x, x_residual)]        # residual again

    # Final projection: 16-dim "thought" vector  ->  27 scores (logits) over the vocabulary.
    # Higher score = model thinks that token is more likely to come next.
    logits = linear(x, state_dict['lm_head'])
    return logits

# %% [markdown]
# ### Peek inside — what does the model predict BEFORE training?
#
# All 4,192 weights were just set to tiny random numbers. Ask the model:
#
# > *"I just fed you BOS at position 0. What's the first letter of a name?"*
#
# Because the weights are random noise, we expect the answer to be essentially meaningless — a roughly **uniform distribution** over all 27 tokens (probability ≈ 1/27 each). This is the baseline. Everything training does to the model can be understood as "make this distribution less uniform, in ways that match the data."
#
# One forward pass takes a couple of seconds (pure-Python, no GPU).

# %%
# ====== DEMO — untrained model's first-letter distribution ======

keys_dbg, values_dbg = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
logits = gpt(BOS, 0, keys_dbg, values_dbg)
probs = [p.data for p in softmax(logits)]

ranked = sorted(enumerate(probs), key=lambda x: -x[1])[:5]
print("UNTRAINED model — top-5 predictions for 'next token after BOS':")
for idx, p in ranked:
    ch = 'BOS' if idx == BOS else uchars[idx]
    bar = '#' * int(p * 400)
    print(f"  {ch:3}   p={p:.4f}   {bar}")

print(f"\nUniform baseline: 1/27 = {1/27:.4f}")
print("All probabilities cluster near this — the untrained model is effectively guessing.")

# %% [markdown]
# ## Step 6 — the training loop
#
# ### What happens per step
#
# 1. **Pick one name** (rotating through `docs`).
# 2. **Tokenize it** with BOS bookends: `"ann"` → `[BOS, a, n, n, BOS]`.
# 3. **Forward pass.** For each position, predict the next token; compute the loss (how wrong we were).
# 4. **Backward pass.** Call `loss.backward()`. The `Value` class walks the computation graph and fills in `.grad` on every one of the 4,192 parameters.
# 5. **Adam update.** Use those gradients to nudge each parameter.
#
# ### The loss: cross-entropy
#
# For each position, we take `-log(probability_the_model_assigned_to_the_correct_next_token)`.
#
# - Confident and right → `log(near 1) ≈ 0` → loss ≈ 0 (no penalty).
# - Confident and wrong → `log(tiny number)` is hugely negative; negated, it is a huge positive — big penalty.
#
# Average across positions, done.
#
# **Units — what is a "nat"?** Because we use the natural logarithm `log()` (base *e*), loss is measured in **nats** — a unit from information theory that counts bits of surprise, but using *e* instead of 2. One nat ≈ 1.44 bits. You do not need to care about the unit itself; you just need one conversion rule, which is the most useful formula in this section:
#
# > **`exp(loss)` tells you the "effective vocabulary size"** the model is choosing from.
#
# Two worked cases:
#
# | loss (nats) | `exp(loss)` | interpretation |
# |---|---|---|
# | `log(27) ≈ 3.296` | `27.0` | the model is essentially uniform over all 27 tokens — pure random guessing |
# | `2.650` | `~14.2` | the model has narrowed its choice to roughly 14 likely tokens per step |
# | `1.000` | `2.72`  | the model is almost certain (picking between ~3 tokens on average) |
# | `0.000` | `1.00`  | perfect prediction — always the correct token |
#
# So when our loss drops from `3.30` → `2.65`, we can translate that directly:
#
# ```
# exp(3.30 - 2.65) = exp(0.65) ≈ 1.92
# ```
#
# The model has become roughly **1.9× more confident** than random. Not world-shaking for a 4,192-parameter toy, but enough to produce plausible-looking names. The *same* formula applies to GPT-4 (typical loss near `~1.5` on natural text → `exp(1.5) ≈ 4.5` equivalent choices per token).
#
# ### The Adam optimizer
#
# Plain gradient descent (`param -= lr * grad`) works, but **Adam** is smarter:
#
# - `m[i]` = running average of recent gradients (**momentum** — keeps heading in a consistent direction).
# - `v[i]` = running average of recent *squared* gradients (per-parameter **adaptive step size** — takes bigger steps where gradients have been small but consistent).
# - Bias correction (`m_hat`, `v_hat`) compensates for the fact that these averages start at zero and need to warm up.
# - `lr_t` linearly decays the learning rate to zero — standard practice, so we fine-tune at the end rather than overshooting.
#
# **Analogy:** plain SGD is rolling a ball down a hill blindfolded. Adam is the same ball, but with a gyroscope (momentum) and shock absorbers calibrated per wheel (per-parameter adaptive steps).
#
# ### What you will see
#
# Loss starts around `log(27) ≈ 3.3` (random guessing among 27 tokens) and drops to roughly `2.6` after 1,000 steps. Not spectacular — this is a 1-layer, 4k-parameter toy — but enough to produce name-like output.
#
# ### Training diagnostics to watch
#
# Every 100 steps, the loop prints a rich snapshot line (separate from the compact per-step progress):
#
# ```
# step  100 | loss 2.9431 | lr 0.00900 | grad_norm 12.34
# ```
#
# - **`loss`** — current training loss. Should trend down.
# - **`lr`** — current learning rate. Linearly decays from `0.01` → `0` over the run.
# - **`grad_norm`** — the "loudness" of the learning signal: `sqrt(sum(grad^2))` across all 4,192 parameters. This is one of the most-watched numbers in real ML training.
#     - **Trending down** → model is converging; gradients shrink as the model gets things right.
#     - **Exploding upward** → unstable training. In a real job you'd halt and lower the learning rate.
#     - **Stuck near zero early** → dead gradients; the model has stopped learning prematurely.
#
# Tracking these three numbers together is how ML engineers diagnose whether a training run is healthy. In a production training job the same three values are logged to Weights & Biases or TensorBoard and monitored on live graphs.
#
# ---
#
# ### Learning Moment — what "Pre-training" actually IS
#
# Everything that happens inside this for-loop IS Stage 1 of Karpathy's pipeline:
#
# - **Next-token-prediction objective** (`-probs[target_id].log()`) — the ONE objective every base model is trained on. No "be helpful", no "be truthful" — just "predict the next token".
# - **Gradient descent over a giant computation graph** — the `.backward()` call plus the Adam update. Real LLMs run this same pattern on thousands of GPUs for weeks.
# - **No human involvement whatsoever.** The model is learning pure statistics of the training distribution. If the internet said `9.11 > 9.9` more often than not, a base model would too.
#
# After the loop finishes: you own a **base model**. It has absorbed the statistics of baby names. It is not an assistant. It does not know it is a model. It just knows "given these tokens, here is what usually comes next". Generating from it (next cell) will confirm this — it produces names, not answers to questions, because no one ever showed it what a question-and-answer looks like.

# %%
# --- Adam optimizer setup ---
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)   # first moment  (EMA of gradients)
v = [0.0] * len(params)   # second moment (EMA of squared gradients)

num_steps = 1000

# Preamble so you know what "good" looks like before the numbers start scrolling.
print(f"Starting training for {num_steps} steps")
print(f"  -> target: loss should drop from ~{math.log(vocab_size):.2f} (random guessing among {vocab_size} tokens) "
      f"toward ~2.6 by step {num_steps}")
print(f"  -> watch: loss trending DOWN, grad_norm staying STABLE (not exploding)\n")

for step in range(num_steps):

    # ---- 1. Prepare one training example ----
    doc = docs[step % len(docs)]                                 # cycle through the dataset
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]    # e.g. [BOS, a, n, n, BOS]
    n = min(block_size, len(tokens) - 1)                         # how many next-token predictions we make

    # ---- 2. Forward pass: predict next token at every position, accumulate loss ----
    # keys/values[li] is the KV cache for layer li. It grows as we step through positions.
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id  = tokens[pos_id]        # the letter we are looking at now
        target_id = tokens[pos_id + 1]    # the letter we SHOULD predict
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        # Cross-entropy: -log(prob assigned to the correct token).
        # Small if the model was confident + correct; huge if it was confident + wrong.
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)   # mean loss over this sequence

    # ---- 3. Backward pass: fill in .grad on every parameter ----
    # This walks the ENTIRE computation graph we just built
    # (embeddings, attention, MLP, softmax, log — every single Value operation above).
    loss.backward()

    # ---- Diagnostics: snapshot gradient norm BEFORE Adam zeros the grads out ----
    # grad_norm = sqrt(sum of every parameter's squared gradient).
    # Intuition: how "loud" is the learning signal right now?
    #   - grad_norm trending down = model is converging (less to correct each step).
    #   - grad_norm exploding     = unstable training (loss would diverge).
    #   - grad_norm ~ 0           = model has stopped learning (good if at end, bad if early).
    grad_norm = sum(p.grad ** 2 for p in params) ** 0.5

    # ---- 4. Adam update: apply gradients to parameters ----
    lr_t = learning_rate * (1 - step / num_steps)   # linear decay to 0 over training
    for i, p in enumerate(params):
        # Update running averages of gradient and squared gradient.
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # Bias-correct (these EMAs start at 0, so they are biased toward 0 early on).
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        # Take a step: direction = m_hat, step size scaled per-param by 1/sqrt(v_hat).
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # Clear the gradient for the next iteration (otherwise it would keep accumulating).
        p.grad = 0

    # Compact progress every step (same line, overwritten by \r)
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

    # Rich diagnostic snapshot every 100 steps (new line, stays on screen as a training trace)
    if (step + 1) % 100 == 0 or step == 0:
        print(f"step {step+1:4d} | loss {loss.data:.4f} | lr {lr_t:.5f} | grad_norm {grad_norm:.4f}")

# Post-training summary so the final loss has meaning.
baseline = math.log(vocab_size)
drop = baseline - loss.data
print(f"\nTraining complete.")
print(f"  -> final loss: {loss.data:.4f}  (started near {baseline:.2f}, the random-guess baseline)")
print(f"  -> loss dropped by {drop:.2f} nats  "
      f"=>  the model now assigns ~{math.exp(drop):.1f}x more probability to real next-letters than chance")
print(f"  -> the 4,192 parameters have been nudged into a useful configuration. Time to generate names.")

# %% [markdown]
# ### Peek inside — what does the model predict AFTER training?
#
# Same question as before — *"first token after BOS?"* — but now with weights shaped by 1,000 training steps. Two things should happen:
#
# 1. The distribution should no longer be uniform. Probability should **concentrate on letters that actually start names** (like `a`, `k`, `m`, `s`) and drop near zero for tokens that don't (BOS itself, for example — names never start with BOS).
# 2. The model's distribution should **approximately match the real first-letter frequency in the dataset**. That is the entire point of pre-training: imitate the statistics of the training data.
#
# We'll print both distributions side by side.

# %%
# ====== DEMO — trained model vs. real-data distribution ======
from collections import Counter

# Trained model's prediction for 'first letter of a name'
keys_dbg, values_dbg = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
logits = gpt(BOS, 0, keys_dbg, values_dbg)
probs = [p.data for p in softmax(logits)]

ranked_model = sorted(enumerate(probs), key=lambda x: -x[1])[:10]
print("TRAINED model — top-10 first-letter predictions:")
for idx, p in ranked_model:
    ch = 'BOS' if idx == BOS else uchars[idx]
    bar = '#' * int(p * 150)
    print(f"  {ch:3}   p={p:.4f}   {bar}")

# Compare to the true first-letter distribution in the dataset
first_letter_counts = Counter(doc[0] for doc in docs)
total = sum(first_letter_counts.values())
print("\nREAL DATA  — top-10 first-letter frequencies:")
for ch, cnt in first_letter_counts.most_common(10):
    p_real = cnt / total
    bar = '#' * int(p_real * 150)
    print(f"  {ch:3}   p={p_real:.4f}   {bar}")

print("\nThe two distributions should roughly match. The model has learned what a name's first letter 'looks like'.")

# %% [markdown]
# ### Peek inside — what is the model paying attention to?
#
# So far the transformer has been a black box. Let's crack it open.
#
# Recall the study-group analogy: at each position, the model asks a **query**, matches it against every past token's **key**, and uses the resulting probabilities (the **attention weights**) to pull information from past tokens. Those weights are how the model decides *what to focus on*.
#
# This demo feeds the prefix `[BOS, e, m, m]` through the model, then asks the last token (`'m'`) to attend to its past. For each of the 4 heads, we print:
#
# - the weight placed on each past position,
# - a bar chart so the distribution is easy to read,
# - the sum of the weights (should be exactly 1.0 — attention is a probability distribution).
#
# **What to look for:** different heads often develop different "personalities". One head might put most of its weight on the immediately-preceding token ("what letter just came?"), another might spread weight broadly ("what kind of name is this?"), another might focus on BOS ("where am I in the name?"). These specialisations emerge from training with no explicit instruction.
#
# This demo needs a slightly modified `gpt()` that returns attention weights alongside logits. We define it here so we do not pollute the original `gpt()`.

# %%
# ====== DEMO — extract and visualise attention weights ======

def gpt_with_attn(token_id, pos_id, keys, values):
    """Identical to gpt(), but also returns per-head attention weights for inspection."""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    attn_by_head = []   # captured per head, per layer

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            attn_by_head.append([w.data for w in attn_weights])   # capture for demo
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits, attn_by_head


# Feed the prefix [BOS, e, m, m] and grab the attention weights at the final position.
prefix_tokens = [BOS, uchars.index('e'), uchars.index('m'), uchars.index('m')]
prefix_labels = ['BOS', 'e', 'm', 'm']

keys_vis, values_vis = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
last_attn = None
for pos_id, tok in enumerate(prefix_tokens):
    _, attn = gpt_with_attn(tok, pos_id, keys_vis, values_vis)
    last_attn = attn   # keep the attention from the LAST forward pass

print(f"Attention weights — what the last token ('{prefix_labels[-1]}' at pos {len(prefix_tokens)-1}) is attending to.")
print(f"Context so far: {prefix_labels}")
print()

for h, weights in enumerate(last_attn):
    print(f"Head {h}:")
    for label, w in zip(prefix_labels, weights):
        bar = '#' * int(w * 50)
        print(f"    -> '{label:3s}'   w={w:.3f}   {bar}")
    print(f"    sum = {sum(weights):.4f}   (must be 1.0 — attention is a probability distribution)")
    print()

print("Compare the 4 heads: each one specialises in a different attention pattern.")
print("These specialisations were NOT programmed — they emerged from training.")

# %% [markdown]
# ## Step 7 — inference (generating new names)
#
# The trained model now has its 4,192 parameters nudged into a useful configuration. Time to let it babble.
#
# ### Greedy vs. sampling
#
# We *could* always pick the single highest-scoring next token, but that produces repetitive, boring output. Instead we **sample** from the probability distribution — with a twist: **temperature**.
#
# ### Temperature
#
# Temperature divides the logits before softmax:
#
# - `temperature == 1.0` → sample from the model's natural distribution.
# - `temperature < 1.0` → **sharpen** the distribution toward the most likely tokens (more conservative, more coherent, less diverse).
# - `temperature > 1.0` → **flatten** the distribution (more creative, more errors).
#
# **Analogy:** temperature is a "boldness" dial. Cold = play it safe. Hot = take risks.
#
# ### Generation loop
#
# Start with the BOS token; feed it in; sample the next token; feed *that* in; repeat. Stop when the model emits BOS again (it is saying "name ended") or we hit `block_size`.
#
# ---
#
# ### The full generation trace — one concrete example
#
# Let's walk through *exactly* what happens when the code produces one name. Suppose the output is `"kamon"`. Here is every iteration, start to finish.
#
# **Setup (before the loop):**
#
# ```python
# token_id = BOS    # the "go" signal — token id 26
# sample   = []     # where real letters will be collected
# keys, values = [[]], [[]]   # KV cache, empty
# ```
#
# **Iteration 0** — `pos_id=0`:
# - Feed `BOS` into `gpt(BOS, 0, keys, values)` → 27 logits.
# - Softmax → 27 probabilities. Because during training the model saw *"BOS → first-letter-of-a-name"* tens of thousands of times, the distribution concentrates on plausible name-starting letters.
# - `random.choices(...)` → say we sample `'k'` (token `10`).
# - `token_id != BOS`, so → `sample.append('k')` → `sample = ['k']`.
#
# **Iteration 1** — `pos_id=1`:
# - Feed `'k'` into `gpt('k', 1, keys, values)`. Attention can now look back at the past (BOS + `'k'`) via the KV cache.
# - Softmax → vowels get most of the probability mass (`k` is usually followed by a vowel in names).
# - Sample `'a'`. `sample = ['k', 'a']`.
#
# **Iterations 2–4** — `pos_id=2,3,4`:
# - Continue feeding the most-recently-sampled token in. The model samples `'m'`, then `'o'`, then `'n'`.
# - `sample = ['k', 'a', 'm', 'o', 'n']`.
#
# **Iteration 5** — `pos_id=5`:
# - Feed `'n'` into `gpt('n', 5, ...)`.
# - The model has now attended to `[BOS, k, a, m, o, n]` — a complete-looking name.
# - During training, every name was wrapped as `[BOS, ..letters.., BOS]`, which made BOS **the target** at the position right after the last real letter. The model's weights therefore encoded: *"after a plausible-looking complete name, BOS is what comes next."*
# - Softmax → most probability now sits on BOS.
# - Sample → `BOS`.
# - `if token_id == BOS: break`. Loop exits **before** `sample.append(...)` runs.
#
# **After the loop:** `''.join(sample)` → `"kamon"`.
#
# ---
#
# ### Three things to notice
#
# **1. BOS is both the "go" signal AND the "stop" signal — same token, two roles.**
#
# When BOS is the **input**, it means *"a name is starting, please generate"*. When BOS is the **sampled output**, it means *"the name is complete, please halt"*. The model does not distinguish between these roles — it just learned from training that BOS appears at both ends of every name. The wrapping `[BOS] + [letters] + [BOS]` at training time **is** what teaches the model this dual role.
#
# **2. Why does the model "know" to emit BOS at the end?**
#
# Look back at the training loop:
#
# ```python
# tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
# ```
#
# For every name, the position right after the last real letter had **BOS as its prediction target**. The model was punished (via cross-entropy loss) every time it failed to predict BOS at that position. After 1,000 steps, its weights encoded: *"when the context looks like a finished name, emit BOS."* At inference, those same weights fire, BOS gets sampled, and we stop.
#
# **No stop-token logic is hard-coded into the model.** Stopping is a pure statistical consequence of how we framed the training data. This is the same mechanism real LLMs use — ChatGPT "knows" when to stop because an `<|endoftext|>` token appears at the end of every training example, so the model learns to emit it once its output looks complete.
#
# **3. Why the starting BOS never appears in the output.**
#
# Look at the code carefully:
#
# ```python
# token_id = BOS                             # start signal, fed IN to the model
# sample = []                                # empty collector — BOS is not added here
# for pos_id in range(block_size):
#     logits = gpt(token_id, pos_id, keys, values)
#     probs = softmax([l / temperature for l in logits])
#     token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
#     if token_id == BOS:
#         break                              # stop signal — never reaches the append below
#     sample.append(uchars[token_id])        # only real letters get appended
# ```
#
# - The **initial BOS** is fed in as `token_id` but never written to `sample`.
# - The **final BOS** triggers `break` *before* `sample.append(...)` runs.
# - So `sample` only ever contains real letters. The bookends exist in the model's internal framing but are hidden from the human-readable output.
#
# The visible output `"kamon"` is just the 5 real-letter tokens between the two invisible BOS bookends.

# %% [markdown]
# ### Peek inside — verbose trace of ONE generation
#
# The walkthrough above is hypothetical. This cell runs a **real, single-sample generation** and prints, at every step:
#
# - which token is being fed in,
# - the top-3 most-probable next tokens with their probabilities,
# - which token got sampled,
# - the accumulated output so far.
#
# You will see the model's distribution shift as context builds up, and — crucially — you'll see the step where **BOS rises to the top of the distribution** and triggers the `break`.
#
# After this demo, the next code cell runs the normal 20-sample generation without the verbose output.

# %%
# ====== DEMO — verbose trace of ONE generation, with top-3 predictions per step ======

random.seed(12345)   # fixed seed so the trace is reproducible for this cell
demo_temperature = 0.5

keys_dbg, values_dbg = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
token_id = BOS
sample = []

print(f"Start:  fed token = BOS (id {BOS}),  sample = []\n")

for pos_id in range(block_size):
    logits = gpt(token_id, pos_id, keys_dbg, values_dbg)
    probs  = softmax([l / demo_temperature for l in logits])
    probs_data = [p.data for p in probs]

    top3 = sorted(enumerate(probs_data), key=lambda x: -x[1])[:3]
    top3_str = '  '.join(f"{('BOS' if i==BOS else uchars[i])}={p:.2f}" for i, p in top3)

    sampled_id = random.choices(range(vocab_size), weights=probs_data)[0]
    fed_char     = 'BOS' if token_id == BOS else uchars[token_id]
    sampled_char = 'BOS' if sampled_id == BOS else uchars[sampled_id]
    so_far = ''.join(sample) if sample else '(empty)'

    print(f"pos {pos_id:2}: fed '{fed_char:3}' | top-3: [{top3_str}] | sampled -> '{sampled_char}' | so far: {so_far}")

    if sampled_id == BOS:
        print("\n  >>> BOS sampled. The model said 'name ended'. Loop breaks here, BOS is NOT appended.")
        break

    sample.append(uchars[sampled_id])
    token_id = sampled_id

print(f"\nFinal name produced: '{''.join(sample)}'")

# %%
temperature = 0.5   # 0 < T <= 1: lower = more conservative. 0.5 works well for names.

print("\n--- inference: generating 20 new names from scratch ---")
print(f"  -> temperature = {temperature}  (lower = safer picks, higher = more creative but more errors)")
print(f"  -> each name is invented letter-by-letter by sampling from the model's learned distribution")
print(f"  -> NONE of these names appear in the training data — they are freshly generated\n")

for sample_idx in range(20):
    # Fresh KV cache per sample (we are generating from scratch each time).
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS   # every sample starts from the BOS signal
    sample = []

    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        # Divide logits by temperature BEFORE softmax to control distribution sharpness.
        probs = softmax([l / temperature for l in logits])
        # Sample one token, weighted by the probabilities.
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break   # model signalled end-of-name
        sample.append(uchars[token_id])

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

# %% [markdown]
# ## What you just built — and what is missing
#
# ### You now have a Stage-1 base model
#
# Look at the 20 samples above. They are plausible-looking *names*. They are not answers to any question. If you somehow "asked" this model *"What is 2 + 2?"* (by feeding in those tokens — except we don't have those tokens in our vocab), it would not answer; it would continue the string in whatever way best fits the statistics of baby names.
#
# This is **exactly** the state real base models (e.g. GPT-3 before ChatGPT) were in. They were brilliant autocompletes, but unusable as assistants.
#
# ---
#
# ### Stage 2 — Supervised Fine-Tuning (SFT): the missing "assistant persona"
#
# **What would we do next if we were building ChatGPT?**
#
# 1. Collect a dataset of **conversations**: human-written `(prompt, ideal assistant reply)` pairs — tens to hundreds of thousands of them.
# 2. Continue training the base model on this dataset with the *same* next-token objective.
# 3. The model's weights drift: instead of imitating random internet text, it now imitates the tone of a specific set of **human labelers** who were told to be *"helpful, truthful, and harmless"*.
#
# **Karpathy's key insight:** after SFT, the model is no longer simulating "the internet". It is simulating *"the average helpful human labeler that OpenAI hired"*. The personality you feel when you chat with Claude or ChatGPT is, literally, a statistical average of that labeler pool.
#
# We do not do this in this notebook — but if you wanted to, the mechanics would be: build a small dataset of `(question, answer)` pairs, tokenize them with a separator token, and run *exactly the same* training loop you just ran.
#
# ---
#
# ### Stage 3 — Reinforcement Learning (RL / RLHF): the missing "reasoning & alignment"
#
# SFT makes the model *sound* like an assistant. RL makes it *reason* like one.
#
# **How it works (conceptually):**
# - For problems with verifiable answers (math, code), let the model generate many solution paths, keep the ones that got the right answer, train on those.
# - For subjective tasks (jokes, essays), train a **Reward Model** on human preference comparisons (*"response A is better than response B"*), then let the base model practice against the reward model.
#
# **Emergent reasoning.** When Stage 3 is done at scale (DeepSeek R1, OpenAI o1, etc.), something remarkable happens: the model spontaneously develops **chains of thought** — long internal monologues where it double-checks its work, backtracks, and corrects itself. This was never explicitly programmed; it emerged because chains of thought *empirically* led to more correct answers, and RL reinforced whatever worked.
#
# We do not do this in this notebook either.

# %% [markdown]
# ## Sharp edges — LLM psychology, visible in this toy
#
# Karpathy calls these the *"Swiss cheese"* properties of LLMs: holes that look surprising from the outside but are obvious once you have built one yourself. This notebook makes several of them literal.
#
# ### 1. Hallucination is not a bug — it IS the algorithm
#
# Look at the inference cell. The single line that matters:
#
# ```python
# token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
# ```
#
# That is the entire "hallucination mechanism". The model does not *know* anything. It samples one token from a probability distribution. Then it feeds that token back in and samples again. Karpathy's phrase: a **statistical token tumbler**.
#
# A real LLM asked "Who was the CEO of Microsoft in 1882?" will confidently invent a name — same reason our model invents `vialan`. It does not have a database; it has weights. Weights tumble out plausible tokens regardless of whether the underlying fact exists.
#
# ### 2. Context window = working memory
#
# `block_size = 16` in our toy. The model can only "see" the last 16 tokens. In a real LLM this is 128k, but still finite. Parameters hold **vague recollection**; the context window holds **sharp, current information**.
#
# **Practical takeaway:** when using a real LLM, paste the source document into the chat rather than relying on the model to "remember" it from training. You are upgrading vague recollection to working memory.
#
# ### 3. The Swiss cheese model
#
# Our toy writes plausible 6-letter names yet could never count the vowels in one of its own outputs — because we never trained it to count, and counting is not a next-token-prediction-friendly task. Real LLMs solve PhD-level physics while miscounting the `r`s in `"strawberry"` for the same reason: the training objective rewards sounding right, not being right.
#
# ### 4. No tools
#
# Our model has **no external tools**. Everything it "knows" lives in 4,192 numbers. Real frontier LLMs are increasingly trained to call out to Python interpreters and web search — precisely to escape the statistical-tumbler limits we just demonstrated. When GPT-4 answers `"what is 847 * 293"` correctly, there is a very good chance it invoked a Python tool rather than trying to compute it from weights.
#
# ---
#
# ### Try it yourself — make the sharp edges visible
#
# After running the notebook:
#
# 1. **Force-generate 200 names, not 20.** Count how often you get a duplicate. Base models love to repeat themselves — a live demonstration of mode collapse / low diversity at low temperature.
# 2. **Crank `temperature` to 2.0.** Watch coherence collapse. Same weights, different sharpness — the "hot" dial in action.
# 3. **Start generation from a non-BOS token** (edit: `token_id = uchars.index('z')`). The base model will cheerfully continue as if this is a normal mid-name — confirming it is a continuation engine, not a question-answering engine.

# %% [markdown]
# ## Exercises — turn the notebook into a workshop
#
# These are ordered roughly by difficulty. Each one exercises a specific concept from a specific step. Answers are not supplied; the *doing* is the point.
#
# ### Beginner — predict before you run
#
# **E1. (Step 4)** Before running the param-demo cell, predict: if you change `n_embd` from `16` to `32`, what's the new total parameter count? Verify by editing the code and rerunning.
#
# **E2. (Step 2)** Without editing code, use the tokenizer demo cell: encode `"anna"` manually on paper, then run the encoder to check. Which token ID is `'a'`? Which is `'z'`?
#
# **E3. (Step 6)** Before training runs, predict what `loss` at step 1 should be close to (hint: `log(vocab_size)` — the random-guess baseline). Check against the actual first printed value.
#
# ### Intermediate — experiment with the model
#
# **E4. (Step 5 / Step 6)** Set `n_layer = 2` and retrain. Does the final loss improve? By how much? Is the improvement worth the ~2× training time?
#
# **E5. (Step 7)** The temperature demo mentioned `0.5` as a good choice. Sweep `[0.1, 0.5, 1.0, 1.5, 2.0]` and generate 10 names at each. Write down what changes qualitatively — diversity vs. coherence.
#
# **E6. (Step 3)** Using only the `Value` class, compute `d/dx (x^3 + 2*x)` at `x = 5`. By hand it's `3*25 + 2 = 77`. Verify by constructing the expression as `Value`s and calling `.backward()`.
#
# **E7. (Step 4 / peek cell 19)** After training completes, rerun the parameter-histogram code from the demo cell. Is the distribution still a neat bell curve, or have the tails grown? The demo cell ends with a prompt inviting you to do exactly this.
#
# ### Advanced — break the model on purpose
#
# **E8. (Step 4)** Change the Gaussian init from `std=0.08` to `std=1.5` (way too big). Retrain. What happens to the loss? To `grad_norm`? You should see activations explode — a live demonstration of why init statistics matter.
#
# **E9. (Step 5)** Remove the ReLU in the MLP block (replace `xi.relu()` with just `xi`). Retrain. Loss should barely improve. Why? Answer: the MLP now collapses into a pure linear transformation (see the "Why do we need a squish?" section) and the model loses all per-token non-linear thinking.
#
# **E10. (Step 6)** Disable the residual connection by changing `x = [a + b for a, b in zip(x, x_residual)]` to `x = [a for a, b in zip(x, x_residual)]` (drop the `+b`). Retrain with `n_layer = 2`. Does loss plateau earlier? Residuals exist specifically to let gradients flow to early layers; removing them should cause observable degradation.
#
# ### Reflective — connect to the bigger picture
#
# **E11. (Cell 34, "What you just built — and what is missing")** Write a 3-sentence explanation, in your own words, of the difference between Stage 1 (pre-training, what this notebook does) and Stage 2 (SFT — supervised fine-tuning, what a real ChatGPT adds on top).
#
# **E12. (Cell 35, "Sharp edges")** The "strawberry problem" (LLMs miscount letters in a word) does NOT apply to our toy. Why not? Hint: think about our tokenizer vs. GPT-4's BPE tokenizer.
#
# **E13. (Step 7 generation trace)** In your own words, explain *why* the model learns to emit BOS at the end of a name, using only the training-loop code as evidence. One paragraph. Do not look at the existing explanation until after you've written yours.

# %% [markdown]
# ## One-page summary — everything you just built, at a glance
#
# ### The 3 stages of an LLM (Karpathy's framework)
#
# | Stage | What it does | Where it appears here |
# |---|---|---|
# | **1. Pre-training** | Learn to predict the next token from raw text | Steps 1–7 of this notebook |
# | **2. SFT** (Supervised Fine-Tuning) | Adopt an assistant persona from human-written examples | not built |
# | **3. RL / RLHF** | Refine with trial-and-error against a reward signal | not built |
#
# ### The 7 steps of Stage 1 (this notebook)
#
# ```
# 1. Dataset           - 32k names   |   real LLMs: ~44 TB text
# 2. Tokenizer         - 27 chars    |   real LLMs: ~100k BPE tokens
# 3. Autograd          - Value class |   real LLMs: PyTorch / JAX
# 4. Parameters        - 4,192       |   real LLMs: up to ~1.8 T
# 5. Transformer       - 1 layer     |   real LLMs: 12 - 120 layers
# 6. Training loop     - 1000 steps  |   real LLMs: billions of steps
# 7. Inference         - 20 samples  |   real LLMs: serve millions of users
# ```
#
# ### The 3 interludes (what they scaffold)
#
# - **Derivatives & gradients** — the math that lets `.backward()` exist.
# - **Neural networks (parts 1 & 2)** — what a neuron, weight, layer, and transformer actually are.
# - **Squish (inside NN interlude)** — why non-linearities are the hinge that makes learning possible.
#
# ### The forward pass for ONE token, in 8 lines
#
# ```
# x = wte[token_id] + wpe[pos_id]           # embed
# x = rmsnorm(x)                            # volume control
# # --- attention block ---
# q = linear(rmsnorm(x), wq); k = ...; v = ...   # project to Q K V
# attn = softmax(q . k_cache / sqrt(d)) . v_cache # weighted mix of past
# x = x + linear(attn, wo)                  # residual
# # --- MLP block ---
# x = x + linear(relu(linear(rmsnorm(x), fc1)), fc2)  # residual
# logits = linear(x, lm_head)               # 27 scores
# ```
#
# That's the entire transformer. Everything else is plumbing.
#
# ### The training loop, in 5 lines
#
# ```
# for step in range(num_steps):
#     loss = mean(cross_entropy(gpt(tokens), targets))    # forward + measure
#     loss.backward()                                     # autograd
#     adam_update(params, grads)                          # learn
#     zero_grads(params)                                  # cleanup
# ```
#
# ### Key numbers to remember
#
# | quantity | value | why it matters |
# |---|---|---|
# | `vocab_size` | 27 | tokens the model knows |
# | `n_embd` | 16 | internal "width" per token |
# | `n_layer` | 1 | depth of the transformer |
# | `n_head` | 4 | parallel attention "perspectives" |
# | `block_size` | 16 | max context window = working memory |
# | params | 4,192 | learnable numbers (the "brain") |
# | loss (start) | ~3.3 nats | random-guess baseline = `log(27)` |
# | loss (end)   | ~2.65 nats | ~1.9x more confident than chance |
# | training     | 1,000 steps | ~2–5 minutes on a laptop |
#
# ### The 5 hard-won ideas you now own
#
# 1. **Tokenization** — a bidirectional lookup table, indexed position ↔ character.
# 2. **Autograd** — every math op drops a breadcrumb; `backward()` walks them applying the chain rule.
# 3. **Weights ARE knowledge** — 4,192 random numbers become a name-generator because of 1,000 tiny nudges each.
# 4. **Attention** — Q · K · V, with the softmax weighting "how much each past token should influence the current one".
# 5. **Stopping is learned, not coded** — BOS emerges as the stop signal because training data wrapped every name in BOS bookends.
#
# ### If you want to go further
#
# - **Make it bigger:** bump `n_layer` / `n_embd`, retrain, see the loss improve.
# - **Swap the dataset:** replace `input.txt` with Shakespeare, code, or anything text. No code changes needed — the tokenizer adapts.
# - **Add Stage 2:** write a few hundred `(question, answer)` pairs, fine-tune. Turn the babbler into an assistant.
# - **Read the real thing:** Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) uses PyTorch but has the same structure as what you just built.

# %% [markdown]
# ## Closing thought
#
# You have now built a full Stage-1 transformer from scratch, end to end. The algorithm you wrote — tokenize → embed → attend → MLP → predict next token → backprop → Adam → repeat — is, without exaggeration, the exact algorithm that produced GPT-4 and Claude and every modern LLM. The only differences are:
#
# - **More data** (44 TB vs. 230 KB)
# - **More parameters** (1.8 T vs. 4 K)
# - **Two extra training stages** (SFT + RL) bolted on top of the pre-training you just ran
#
# Karpathy's final framing is worth repeating: **LLMs are magical but stochastic tools**. They are token tumblers weighted by the statistics of their training data, with a post-hoc persona glued on by Stage 2 and some reasoning habits reinforced by Stage 3. Use them for inspiration and first drafts; verify anything that matters. They are simulations, not infallible beings.
#
# You have now seen the machinery first-hand. The magic is gone — and in its place, something better: understanding.
