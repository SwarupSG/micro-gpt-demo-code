# NOTICE — Intent of the License Split

This repository is licensed in two parts:

- The **Jupytext teaching demos** authored by Swarup Biswas are under the **PolyForm Noncommercial License 1.0.0** — see [LICENSE](LICENSE).
- **Karpathy's original code and dataset** are under **MIT** — see [LICENSE-MIT](LICENSE-MIT).

This NOTICE is a plain-English explanation of what that means for the specific files in this repo. The legal text in `LICENSE` and `LICENSE-MIT` governs in case of any conflict.

---

## What this repo is

A classroom-projection version of the microGPT teaching material. Three Python files progress from Karpathy's canonical `microgpt.py` to a heavily-annotated long-form Jupytext walk-through to a slim live-demo version.

For the deeper seven-module pedagogical scaffold (one module per concept — tokenizer, arithmetic, autograd, forward pass, training step, training loop, inference), see the sibling repo: <https://github.com/SwarupSG/how-llm-work-microgpt>.

---

## File-by-file license

| File | License | Authored by |
|---|---|---|
| `microgpt_original.py` | MIT — see note below | Andrej Karpathy |
| `input.txt` | MIT — see note below | Andrej Karpathy (from `makemore`) |
| `microgpt.py` (long-form Jupytext demo) | PolyForm Noncommercial 1.0.0 | Swarup Biswas |
| `microgpt_demo.py` (slim Jupytext demo) | PolyForm Noncommercial 1.0.0 | Swarup Biswas |

### About the MIT terms on Karpathy's files

Karpathy's `microgpt.py` gist itself has no explicit license. However, his other public GPT-related work — [`nanoGPT`](https://github.com/karpathy/nanoGPT), [`makemore`](https://github.com/karpathy/makemore), and [`micrograd`](https://github.com/karpathy/micrograd) — is MIT-licensed. This repo includes MIT terms (in [LICENSE-MIT](LICENSE-MIT)) for `microgpt_original.py` and `input.txt` consistent with that pattern, and disclaims any restriction this repo would otherwise place on those files.

- `input.txt` originates from Karpathy's [`makemore`](https://github.com/karpathy/makemore) project, which IS explicitly MIT-licensed at the source.
- `microgpt_original.py` is byte-identical to Karpathy's [public gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). If Karpathy publishes the gist under a different license in the future, that license governs `microgpt_original.py`, not the MIT terms in this repo's `LICENSE-MIT`.

If you want Karpathy's code on its own terms, take it from his gist or `makemore` repo directly — not from this teaching scaffold.

---

## What the PolyForm license covers (the demo files)

The Jupytext teaching demos — `microgpt.py` and `microgpt_demo.py` — are pedagogical scaffolding the author built around Karpathy's algorithm.

**You ARE allowed to:**

- Clone, fork, and study these files for personal learning
- Modify them on your own fork for experimentation or coursework
- Reference the techniques in any work, commercial or not — *what you learn is yours*
- Run a free community workshop using these files, with attribution
- Share the repo URL freely

**You ARE NOT allowed to:**

- Sell access to these demo files as part of a paid course, bootcamp, or training programme
- Bundle them into a commercial product (paid course platform, paid certification, commercial training offering)
- Use them as the curriculum for a fee-charging educational offering — whether one-on-one tutoring, a bootcamp, a corporate workshop, or a paid online course
- Re-license derivatives of the demo files under terms that permit the above

(These restrictions are about the *teaching demos*, not Karpathy's `microgpt_original.py` or `input.txt` — see the MIT carve-out above.)

For commercial licensing of the demo files, contact: swarup.biswas@outlook.com

---

## The grey area: internal corporate training

If a company uses this repo to train its own employees internally (no fee charged to trainees, no commercial product built on top), the author considers that **acceptable** under the spirit of the noncommercial license. The PolyForm noncommercial definition is *"primarily intended for or directed toward commercial advantage or monetary compensation"* — internal training without monetization fits that description.

If you're unsure whether your use case falls inside or outside the noncommercial line, **ask** before using it commercially.

---

## Attribution

When you fork, share, or reference this repo, please credit:

- **Andrej Karpathy** — for the canonical `microgpt.py` algorithm and the `makemore` names dataset
- **Swarup Biswas** — for the Jupytext long-form and slim classroom demos

---

## Contact

For commercial licensing inquiries: swarup.biswas@outlook.com
