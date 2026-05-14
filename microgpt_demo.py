# %% [markdown]
# # microGPT — a language model in ~200 lines of pure Python
#
# A tiny neural network that invents plausible baby names — `kamon`, `jaire`, `areli` — after reading 32,000 real names. **Dependency-free**: no PyTorch, no NumPy. Every concept (autograd, attention, optimizer) is built by hand and readable top to bottom.
#
# **The whole thing in three ideas:**
#
# 1. **Forward pass.** Feed a letter → model predicts the next letter.
# 2. **Measure wrongness.** Compare to the real next letter → a single **loss** number.
# 3. **Nudge the knobs.** Adjust the model's internal numbers so it would have been less wrong. Repeat.
#
# Credits: @karpathy's `micrograd` / `nanoGPT`.

# %% [markdown]
# ## Where microGPT fits in the full LLM pipeline
#
# Chat models are built in **three stages**. We build Stage 1 only.
#
# - **Stage 1 — Pre-training.** Predict the next token, billions of times. Output = "base model" / internet-document simulator. **← built below.**
# - **Stage 2 — SFT.** ~100k (prompt, ideal reply) pairs → the helpful-assistant persona. *Skipped.*
# - **Stage 3 — RL/RLHF.** Reward good solutions; chains-of-thought emerge. *Skipped.*
#
# | Stage 1 step | Cell | Real-world scale | Our toy |
# |---|---|---|---|
# | Data + filter | Step 1 | FineWeb **~44 TB** | 32k names, ~230 KB |
# | Tokenization | Step 2 | GPT-4 BPE, **~100k tokens** | char-level, **27** |
# | Network | Steps 4–5 | hundreds of **billions** of params | **4,192** |
# | Training | Step 6 | trillions of tokens, weeks, ×1000 GPUs | 1,000 steps, laptop |
# | Result | Step 7 | "Internet simulator" | "Name simulator" — same algorithm |
#
# > Deep dive: `microgpt_teaching_walkthrough.md` §pipeline.

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
# 32,033 lowercase baby names from Karpathy's `makemore`, char-level, no GPU needed. Two filters below mirror what real pipelines do at scale: `line.strip()` drops empty lines (real pipelines drop spam/PII/malware/duplicates), and `random.shuffle` removes ordering bias (real pipelines also re-weight by quality). Same ideas, different scale.

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
# **A translator between text and numbers** — networks only do arithmetic. Both directions share ONE sorted list (`uchars`), and **the index IS the token ID**:
#
# - **Encode** `'n'` → `uchars.index('n')` → `13`
# - **Decode** `13` → `uchars[13]` → `'n'`
#
# For our toy: 26 letters `a..z` (IDs 0–25) + one extra **BOS** token (ID 26) as a bookend — *"name starts here"* and *"name ends here"*.
#
# Example: `"ann"` → `[BOS, a, n, n, BOS]` → `[26, 0, 13, 13, 26]`.
#
# **Our vocab = 27.** GPT-4 BPE vocab ≈ **100,000** — common chunks like `" the"` are one token. Two reasons real LLMs use BPE: shorter sequences (transformer cost is quadratic in length) and better generalisation (one token for `"running"` vs. reconstructing from 7 letters every time).
#
# **Sharp edge — the "strawberry problem".** Because real LLMs see `"strawberry"` as 2–3 BPE tokens, not 10 letters, they miscount letters inside it. Same reason they think `9.11 > 9.9` (tokenized as `9 . 11` vs `9 . 9`). Our char-level model wouldn't — but it loses the chunk-level shortcuts.
#
# > Deep dive: `microgpt_teaching_walkthrough.md` §tokenizer; `how-llm-work-microgpt/01_tokenizer/README.md`.

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
# ## Interlude — derivatives, gradients, autograd in six bullets
#
# - **Derivative** = slope. *"Nudge input by δ; how much does output move?"* It's the answer to "should I turn this knob up or down?"
# - **Gradient** = the vector of slopes, one per parameter. 4,192 derivatives, one per knob. Points uphill; `-gradient` points downhill into lower loss. (Blindfolded hiker on a 4,192-D mountain.)
# - **Chain rule** = relay race of slopes. Each operation passes a "sensitivity baton" via its local derivative; slopes multiply along the chain to the loss.
# - **Autograd** = record-and-replay. Every `+`, `*`, `log` knows its local derivative. We record the graph as it's built, then walk it backwards applying the chain rule.
# - **`.backward()`** topologically sorts the graph then deposits `local_grad * v.grad` onto every child.
# - **Stored rules in the code:** `(1, 1)` for add, `(other.data, self.data)` for multiply — those tuples ARE the derivative rules.
#
# > Deep dive: `neural_networks_gpt_fundamentals.md` §training mechanism; `how-llm-work-microgpt/03_autograd/README.md`.

# %% [markdown]
# ## Step 3 — autograd: a number that remembers its own history
#
# Every number wrapped in `Value`. Each `Value` knows four things:
# - **`data`** — the float (forward pass).
# - **`grad`** — `∂loss/∂self`, filled in by backward pass.
# - **`_children`** — which `Value`s it was built from (the computation graph).
# - **`_local_grads`** — local derivative w.r.t. each child (the chain-rule link).
#
# Forward: operator overloading (`+`, `*`, …) builds the graph as a side effect of doing math. Backward: walk graph in reverse, deposit gradients via chain rule. The `(other.data, self.data)` you'll see in `__mul__` IS the derivative rule for multiplication.

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
# ### Peek inside — autograd on a problem verifiable by hand
#
# `c = a*b + a` with `a=2, b=3`. By hand: `c = 8`, `dc/da = b + 1 = 4`, `dc/db = a = 2`. Autograd should produce the same numbers.

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
# ## Interlude — what is a neural network, really?
#
# Four ideas every transformer reuses:
#
# - **Neuron** = `squish(w1·x1 + w2·x2 + … + wn·xn)`. The weights = how much this little judge cares about each input.
# - **Squish** = a function with a bend (here `relu(x) = max(0, x)`). Without it, stacking layers collapses to a single matrix multiply — depth gains nothing.
# - **Layer** = many neurons in parallel = one matrix multiply (the `linear(x, w)` function below).
# - **Weights ARE the knowledge.** Knowledge is distributed across millions of weights; no individual one "knows" anything. *"Paris is in France"* in GPT-4 is a statistical residue smeared over many weights.
#
# > Deep dive: `neural_networks_gpt_fundamentals.md` (ReLU/GELU, neurons); slides §neural-net-fundamentals.

# %% [markdown]
# ## How learning happens, and how a transformer is shaped
#
# **Learning** = three steps repeated millions of times: forward → loss → backward (`.backward()` from the cell above) → nudge every weight by `-grad × learning_rate`.
#
# **A transformer** is a specific layer pattern: **attention** (every position looks at every other position) followed by an **MLP** (per-position thinking), alternating for `n_layer` blocks. That's the whole "secret". This notebook has 1 such block.
#
# > Deep dive: `microgpt_teaching_walkthrough.md` §how-learning-happens.

# %% [markdown]
# ## Step 4 — the parameters (the "brain" of the model)
#
# Every number the model *learns* lives in `state_dict`. Start random; training sculpts them. **5 hyperparameters control every matrix shape:** `n_layer=1`, `n_embd=16`, `block_size=16`, `n_head=4`, `head_dim=4` (= `n_embd/n_head`). `n_head` must divide `n_embd` evenly.
#
# **Where the 4,192 parameters come from:**
#
# | bucket | matrices | count |
# |---|---|---:|
# | Embeddings | `wte` (27×16) + `wpe` (16×16) | **688** |
# | Attention (per layer) | `wq + wk + wv + wo`, each 16×16 | **1,024** |
# | MLP (per layer) | `fc1` (64×16) + `fc2` (16×64) | **2,048** |
# | Output head | `lm_head` (27×16) | **432** |
# | **Total** | `688 + 3,072 × n_layer + 432` | **4,192** |
#
# **Scale check — how small is 4,192, really?**
#
# | model | params | ratio |
# |---|---:|---:|
# | microGPT (this notebook) | **4,192** | 1× |
# | GPT-2 small | 124 M | ~30,000× |
# | GPT-3 | 175 B | ~42 M× |
# | GPT-4 (est.) | 1.8 T | ~430 M× |
#
# Doubling `n_embd` ~4× the params (most matrices are `n_embd × n_embd`); doubling `n_layer` doubles the per-layer cost.
#
# > Deep dive: `microgpt_teaching_walkthrough.md` §parameters; `how-llm-work-microgpt/04_forward_pass/README.md`.

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
# Print every matrix shape (should sum to 4,192) + the 16 numbers currently defining the "personality" of letter `'a'` (= `wte[0]`) + init stats (min/max/mean/std).

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
# Given one token + position (and a cache of past keys/values), `gpt()` returns 27 scores. **The pipeline:**
#
# ```
# token_id, pos_id
#    -> wte[token] + wpe[pos]    "what letter am I, where am I?"  -> 16-vec
#    -> rmsnorm                  rescale to ~unit magnitude (volume control)
#    -> Attention block          look at the past, decide what's relevant now
#    -> MLP block                per-token non-linear think (expand 4x, ReLU, squeeze)
#    -> lm_head                  project to 27 logits
# ```
#
# **Attention — study-group analogy.** Each position writes a **Query** (*what am I looking for?*), every position holds up a **Key** (*here's what I know*) and a **Value** (*here's what I'd tell you*). Current position dots its Q against everyone's K, softmaxes the similarities, takes the weighted sum of Vs. Done — past has been "attended" to.
#
# - **Multi-head** = 4 study groups in parallel on 4-dim slices; outputs concatenated.
# - **Causal masking** = implicit in our code: we only append K/V up to the current position, so there are no future K/V to attend to.
# - **KV cache** = K and V get appended each call; Q is recomputed only for the new token. Real-scale speed hack.
# - **MLP** = attention gathers info, MLP processes it.
# - **Residuals** = `output + input` at each block — gradient highway, lets depth work.
#
# **Same architecture as GPT-4, different scale:**
#
# | Knob | microGPT | GPT-2 small | GPT-3 | GPT-4 (est.) |
# |---|---|---|---|---|
# | `n_layer` | **1** | 12 | 96 | ~120 |
# | `n_embd` | **16** | 768 | 12,288 | ~18,000 |
# | `n_head` | **4** | 12 | 96 | ~128 |
# | `block_size` | **16** | 1,024 | 2,048 | 128,000 |
# | params | **~4 K** | 124 M | 175 B | ~1.8 T |
#
# > Deep dive: `microgpt_teaching_walkthrough.md` §attention, §rmsnorm; slide deck §transformer-block.

# %% [markdown]
# ### Peek inside — attention, on actual numbers
#
# Pure-Python arithmetic, no `Value` objects. Context = `[BOS, 'e', 'm']`, 4-dim states (16 in the real model). You'll watch:
#
# 1. Current Query, three past Keys, three past Values.
# 2. Raw similarity = `Q·K` per past token → scaled → softmaxed → weights sum to 1.
# 3. Output = weighted sum of Values. The **entire** attention mechanism.

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
# **What happens per step:**
#
# 1. Pick one name (rotating through `docs`).
# 2. Tokenize with BOS bookends: `"ann"` → `[BOS, a, n, n, BOS]`.
# 3. **Forward** — predict next token at each position; compute loss.
# 4. **Backward** — `loss.backward()` fills in `.grad` on every one of the 4,192 parameters.
# 5. **Adam update** — nudge each parameter by `-lr × m_hat / sqrt(v_hat)`.
#
# **Loss = cross-entropy** = `-log(probability assigned to correct next token)`. Measured in **nats** (natural log). Key rule:
#
# > **`exp(loss)` = the effective vocab size the model is choosing from.**
#
# | loss | `exp(loss)` | interpretation |
# |---:|---:|---|
# | `log(27) ≈ 3.30` | 27.0 | uniform over all tokens — random guessing |
# | `2.65` | ~14 | narrowed to ~14 likely tokens per step |
# | `1.00` | 2.72 | almost certain (~3 plausible options) |
# | `0.00` | 1.00 | perfect prediction |
#
# **Adam** (vs plain SGD): `m` = momentum (EMA of grads), `v` = adaptive step (EMA of grad²), `lr_t` linearly decays. Rolling ball with gyroscope + per-wheel shock absorbers.
#
# **What to watch in the diagnostic line** (`step | loss | lr | grad_norm`):
# - **loss** trending down = converging.
# - **lr** linearly decays `0.01 → 0`.
# - **grad_norm** stable = healthy; exploding = unstable (lower lr); dead-zero = stuck.
#
# > Deep dive: `how-llm-work-microgpt/06_training_loop/README.md`.

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
# Feed `[BOS, e, m, m]` and print, per head, how much weight position-3 (`'m'`) puts on each past position.
#
# **What to look for:** different heads develop different "personalities" — one weights the immediately-preceding token, another spreads broadly, another focuses on BOS. These specialisations emerged from training, not programming.

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
# - **Greedy** (pick the top token every step) → repetitive, boring. So we **sample** from the probability distribution.
# - **Temperature** divides the logits before softmax. `T=1` = natural; `T<1` = sharpened (conservative); `T>1` = flatter (creative, more errors). A boldness dial.
# - **Loop**: start with BOS, sample next token, feed it back in, repeat until the model emits BOS again or we hit `block_size`.
#
# **Sharp insight — BOS is both "go" AND "stop", same token.** When BOS is the **input**, it means "name starting, please generate." When BOS is **sampled**, the loop's `if token_id == BOS: break` halts. The model learns this dual role purely from training data: every name was wrapped `[BOS] + letters + [BOS]`, so the model was punished whenever it failed to predict BOS at the position right after a complete-looking name. The same mechanism makes ChatGPT stop — `<|endoftext|>` learned, not hard-coded.
#
# > Deep dive: `how-llm-work-microgpt/07_inference/README.md`.

# %% [markdown]
# ### Peek inside — verbose trace of ONE generation
#
# Per step: token fed in, top-3 predictions with probabilities, token sampled, output so far. Watch the distribution shift as context builds, then **BOS rises** and triggers `break`.

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
# You now have a **Stage-1 base model**. It produces names, not answers. Same state real base models (GPT-3 before ChatGPT) were in — brilliant autocompletes, unusable as assistants.
#
# - **Stage 2 — SFT.** Continue training on ~100k human-written `(prompt, ideal reply)` pairs. Same loop, different data. The model's "personality" you feel chatting with Claude/ChatGPT is a statistical average of OpenAI's labeler pool.
# - **Stage 3 — RL/RLHF.** For verifiable tasks (math, code): generate many solutions, train on the ones that worked. For subjective tasks: train a reward model on human preference comparisons. At scale this is where **chains of thought** emerge — never programmed, just reinforced because they empirically led to correct answers.
#
# > Deep dive: `microgpt_teaching_walkthrough.md` §closing; closing slides §Stage-2-3.

# %% [markdown]
# ## Sharp edges — LLM psychology, visible in this toy
#
# Karpathy's *Swiss-cheese* model: holes that look surprising from outside but are obvious once you've built one yourself.
#
# ### 1. Hallucination is not a bug — it IS the algorithm
#
# ```python
# token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
# ```
#
# - This single line is the entire "hallucination mechanism". A **statistical token tumbler**.
# - The model does not *know* anything. It samples plausible-next-token. Repeat.
# - A real LLM invents a CEO of Microsoft in 1882 for the same reason ours invents `vialan` — weights, not facts.
#
# ### 2. Context window = working memory
#
# - `block_size = 16` here, 128k in GPT-4 — still finite.
# - Parameters = **vague recollection** (compressed during training). Context = **sharp working memory** (ephemeral, exact).
# - Practical takeaway: paste the source doc *into* the chat. Upgrades vague recollection to working memory.
#
# ### 3. Swiss cheese — competent at hard things, fails at easy ones
#
# - Our toy writes plausible names but couldn't count vowels in its own output — we never trained it to count.
# - Same reason real LLMs solve PhD physics yet miscount `r`s in "strawberry": the objective rewards sounding right, not being right.
#
# ### 4. No tools
#
# - Everything our model "knows" lives in 4,192 numbers — no calculator, no search.
# - Frontier LLMs increasingly call Python interpreters and web search to escape this limit. GPT-4 answering `847 × 293` likely invoked a tool.

# %% [markdown]
# ## Exercises — turn the notebook into a workshop
#
# Ordered by difficulty. Answers not supplied — the doing is the point.
#
# ### Beginner — predict before you run
#
# - **E1 (Step 4)** Predict new param count if `n_embd: 16 → 32`. Verify by editing and rerunning.
# - **E2 (Step 2)** Encode `"anna"` by hand, then check against the encoder. What's `'a'`? What's `'z'`?
# - **E3 (Step 6)** Predict step-1 loss before training runs (hint: `log(vocab_size)`). Check the first printed line.
#
# ### Intermediate — experiment
#
# - **E4 (Step 5/6)** Set `n_layer = 2` and retrain. Loss improvement worth the ~2× compute?
# - **E5 (Step 7)** Sweep `temperature` over `[0.1, 0.5, 1.0, 1.5, 2.0]` × 10 samples each. Diversity vs. coherence?
# - **E6 (Step 3)** Compute `d/dx (x³ + 2x)` at `x=5` (by hand = 77) using only the `Value` class.
# - **E7 (Step 4)** Rerun the parameter histogram after training. Still a bell curve, or tails grown?
#
# ### Advanced — break it on purpose
#
# - **E8 (Step 4)** `std=1.5` init (way too big). Watch loss and `grad_norm` explode. Why init scale matters.
# - **E9 (Step 5)** Remove ReLU in the MLP (`xi` instead of `xi.relu()`). Loss barely improves. Why? The MLP collapses to linear.
# - **E10 (Step 6)** Drop the `+ b` from the residual connection. Retrain with `n_layer = 2`. Earlier plateau, because gradients can't reach early layers.

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
# You just wrote the exact algorithm that produced GPT-4 and Claude. The only differences:
#
# - **More data** (44 TB vs. 230 KB)
# - **More parameters** (1.8 T vs. 4 K)
# - **Two extra training stages** (SFT + RL) bolted on top
#
# LLMs are **statistical token tumblers** weighted by training data, with a persona glued on by Stage 2 and reasoning habits reinforced by Stage 3. Magical, stochastic, useful for first drafts — verify anything that matters.
