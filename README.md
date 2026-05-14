# microGPT — Jupytext classroom demo

Three Python files for projecting [microGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — Andrej Karpathy's tiny dependency-free GPT — through a live classroom session or self-paced study, cell by cell, in an IDE.

By the end of any of these files you will have built a tiny but real GPT that learns to generate names. The same math runs inside ChatGPT, Claude, Gemini — they differ in scale, data, and engineering, not in algorithm.

---

## What's in this repo

| File | Lines | Purpose |
|---|---:|---|
| [`microgpt_original.py`](microgpt_original.py) | 200 | **Karpathy's canonical script**, byte-identical copy of his [public gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The complete algorithm in one straight read — no cells, no prose, no scaffolding. |
| [`microgpt.py`](microgpt.py) | ~1,800 | **Long-form Jupytext demo.** Same algorithm split into `# %%` cells for cell-by-cell IDE execution, with extensive prose explaining every concept (autograd, attention, training loop, inference). For self-paced study. |
| [`microgpt_demo.py`](microgpt_demo.py) | ~1,100 | **Slim Jupytext demo.** Same code, prose trimmed for live classroom projection — no markdown cell exceeds one screen height. The instructor narrates the depth; this file carries the *what*. |
| [`input.txt`](input.txt) | 32,033 lines | Dataset — baby names from Karpathy's [`makemore`](https://github.com/karpathy/makemore). |
| [`microgpt_anatomy.excalidraw`](microgpt_anatomy.excalidraw) | — | **Visual map** of `microgpt_original.py`: 7 modules in a vertical pipeline, with inputs, outputs, and the 20 real names the script produces with `seed=42`. Open in [excalidraw.com](https://excalidraw.com/) or the VS Code Excalidraw plugin. |

All three Python files run the same algorithm end to end (data → tokenize → autograd → forward pass → training loop → inference) and produce the same generated names on the same seed.

---

## How to run

**Prerequisites:** Python 3.10+, [VS Code](https://code.visualstudio.com/) with the [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions (or any IDE that understands Jupytext `# %%` cells).

```bash
git clone https://github.com/SwarupSG/micro-gpt-demo-code.git
cd micro-gpt-demo-code
```

**For `microgpt_original.py`** (the canonical Karpathy file) — just run it:

```bash
python3 microgpt_original.py
```

It will train for 1,000 steps (a few minutes on a laptop) and then print 20 generated names.

**For `microgpt.py` or `microgpt_demo.py`** (the cell-by-cell demos) — open the file in VS Code. A `Run Cell` codelens appears above every `# %%` marker; click it, or place the cursor in a cell and press `Shift+Enter` (or `Cmd+Enter` on macOS). **Run the first code cell first** — it contains the imports and seed — then run each subsequent cell in order.

`input.txt` ships with the repo, so the dataset cell will read it from disk without hitting the network. (This avoids a common macOS SSL-cert pitfall where `urllib` fails to download HTTPS on a fresh Python install.)

---

## License

This repo is dual-licensed. See [NOTICE.md](NOTICE.md) for the plain-English explanation.

| | License |
|---|---|
| `microgpt_original.py`, `input.txt` (Karpathy's work) | **MIT** — see [LICENSE-MIT](LICENSE-MIT) |
| `microgpt.py`, `microgpt_demo.py` (Jupytext teaching demos) | **PolyForm Noncommercial 1.0.0** — see [LICENSE](LICENSE) |

**Short version:** clone, fork, study, modify for personal learning, use the techniques in your own work — all fine. Selling these demo files as part of a paid course, bootcamp, or training program is not — contact the author for a commercial licence.

---

## Related repo

For a deeper, seven-module pedagogical scaffold — one module per concept (tokenizer → arithmetic → autograd → forward pass → training step → training loop → inference), each with its own `module.py` cell-by-cell file and per-module README — see [`how-llm-work-microgpt`](https://github.com/SwarupSG/how-llm-work-microgpt).

That sibling repo is the *deep dive*; this repo is the *projection-ready demo*.

---

## Credit

- **Andrej Karpathy** — for the canonical `microgpt.py` algorithm and the `makemore` baby-name dataset. All credit for the underlying GPT goes here.
- **Swarup Biswas** — for the Jupytext long-form and slim classroom demos in this repo (`microgpt.py`, `microgpt_demo.py`).
