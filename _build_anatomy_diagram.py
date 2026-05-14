"""
Build the microgpt anatomy diagram and emit:
  1. microgpt_anatomy.excalidraw — opens in excalidraw.com / VS Code Excalidraw plugin
  2. /tmp/microgpt_anatomy_compact.json — the elements-only form for create_view preview

The diagram is a 7-module vertical pipeline with input chips on the left,
output chips on the right, and a final "20 generated names" callout.
"""
import json
import random
from pathlib import Path

random.seed(7)  # deterministic versionNonces/seeds in element JSON

HERE = Path(__file__).parent
OUT_FILE = HERE / "microgpt_anatomy.excalidraw"
COMPACT_FILE = Path("/tmp/microgpt_anatomy_compact.json")

elements: list[dict] = []


# ---------- helpers ----------

def rnd():
    return random.randint(1, 2_000_000_000)


def base(eid: str) -> dict:
    """Fields every Excalidraw file element needs."""
    return {
        "id": eid,
        "seed": rnd(),
        "version": 1,
        "versionNonce": rnd(),
        "isDeleted": False,
        "groupIds": [],
        "frameId": None,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
        "roundness": None,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
    }


def rect(eid, x, y, w, h, *, fill=None, stroke=None, label=None,
         label_font=18, rounded=True, label_color="#1e1e1e",
         stroke_width=2, opacity=100):
    """A rounded rectangle, optionally with an auto-centered label."""
    el = base(eid) | {
        "type": "rectangle",
        "x": x, "y": y, "width": w, "height": h,
        "strokeWidth": stroke_width,
        "opacity": opacity,
    }
    if fill is not None:
        el["backgroundColor"] = fill
    if stroke is not None:
        el["strokeColor"] = stroke
    if rounded:
        el["roundness"] = {"type": 3}
    elements.append(el)
    if label is not None:
        tid = f"{eid}_t"
        el["boundElements"] = [{"type": "text", "id": tid}]
        elements.append(base(tid) | {
            "type": "text",
            "x": x, "y": y + h / 2 - label_font * 0.6,
            "width": w, "height": label_font * 1.2,
            "text": label,
            "fontSize": label_font,
            "fontFamily": 5,  # Excalifont (modern Excalidraw default)
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": eid,
            "originalText": label,
            "lineHeight": 1.25,
            "baseline": int(label_font * 0.85),
            "strokeColor": label_color,
        })


def text(eid, x, y, txt, *, font=16, color="#1e1e1e", align="left", bold=False):
    """A standalone text element (titles, annotations)."""
    width = max(80, int(len(txt) * font * 0.55))
    elements.append(base(eid) | {
        "type": "text",
        "x": x, "y": y, "width": width, "height": int(font * 1.4),
        "text": txt,
        "fontSize": font,
        "fontFamily": 5,
        "textAlign": align,
        "verticalAlign": "top",
        "originalText": txt,
        "lineHeight": 1.25,
        "baseline": int(font * 0.85),
        "strokeColor": color,
    })


def arrow(eid, x1, y1, x2, y2, *, color="#1e1e1e", width=2,
          start_id=None, end_id=None, label=None, label_font=14, dashed=False):
    """A straight arrow from (x1,y1) to (x2,y2), optionally bound to shapes."""
    dx, dy = x2 - x1, y2 - y1
    el = base(eid) | {
        "type": "arrow",
        "x": x1, "y": y1, "width": abs(dx), "height": abs(dy),
        "points": [[0, 0], [dx, dy]],
        "lastCommittedPoint": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "strokeColor": color,
        "strokeWidth": width,
        "strokeStyle": "dashed" if dashed else "solid",
    }
    if start_id:
        el["startBinding"] = {"elementId": start_id, "focus": 0, "gap": 4}
    if end_id:
        el["endBinding"] = {"elementId": end_id, "focus": 0, "gap": 4}
    elements.append(el)
    if label:
        tid = f"{eid}_t"
        el["boundElements"] = [{"type": "text", "id": tid}]
        elements.append(base(tid) | {
            "type": "text",
            "x": x1 + dx / 2 - 30, "y": y1 + dy / 2 - 10,
            "width": 60, "height": label_font * 1.2,
            "text": label,
            "fontSize": label_font,
            "fontFamily": 5,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": eid,
            "originalText": label,
            "lineHeight": 1.25,
            "baseline": int(label_font * 0.85),
            "strokeColor": "#555555",
        })


# ---------- colours ----------

BLUE   = "#a5d8ff"   # algorithm modules 1-5
ORANGE = "#ffd8a8"   # training loop (module 6)
PURPLE = "#d0bfff"   # inference (module 7)
YELLOW = "#fff3bf"   # input chips
TEAL   = "#c3fae8"   # output chips
GREEN  = "#b2f2bb"   # final output callout (20 names)
INK    = "#1e1e1e"
SUBTLE = "#757575"


# ---------- title strip ----------

text("title", 320, 20, "microgpt.py  —  anatomy", font=28, color=INK)
text("sub", 215, 60,
     "200 lines of pure Python  ·  dependency-free  ·  dataset  ->  trained model  ->  20 hallucinated names",
     font=15, color=SUBTLE)


# ---------- module column ----------

MOD_X, MOD_W, MOD_H, MOD_GAP = 420, 360, 70, 25
MOD_TOP = 100

modules = [
    ("m1", "1.  Dataset    (load + shuffle)",          BLUE),
    ("m2", "2.  Tokenizer    (uchars + BOS)",          BLUE),
    ("m3", "3.  Autograd    (Value class)",            BLUE),
    ("m4", "4.  Parameters    (state_dict)",           BLUE),
    ("m5", "5.  Model    (gpt function)",              BLUE),
    ("m6", "6.  Training loop    (1000 steps)",        ORANGE),
    ("m7", "7.  Inference    (T = 0.5)",               PURPLE),
]

for i, (mid, label, fill) in enumerate(modules):
    y = MOD_TOP + i * (MOD_H + MOD_GAP)
    rect(mid, MOD_X, y, MOD_W, MOD_H,
         fill=fill, stroke=INK, label=label, label_font=18)


# ---------- arrows between modules ----------

for i in range(len(modules) - 1):
    y_top = MOD_TOP + (i + 1) * (MOD_H + MOD_GAP) - MOD_GAP   # bottom of box i
    y_bot = MOD_TOP + (i + 1) * (MOD_H + MOD_GAP)             # top of box i+1
    arrow(f"chain_{i}", MOD_X + MOD_W / 2, y_top, MOD_X + MOD_W / 2, y_bot,
          start_id=modules[i][0], end_id=modules[i + 1][0])


# ---------- input chips (left column) ----------

def input_chip(eid, target_id, target_y, lines, *, chip_w=180):
    """Yellow chip on the left, with an arrow pointing right into target module."""
    cy = target_y + (MOD_H - 30 - 14 * (len(lines) - 1)) / 2
    chip_h = 18 + 14 * len(lines)
    rect(eid, 220, cy, chip_w, chip_h, fill=YELLOW, stroke="#f59e0b", rounded=True, label=None, stroke_width=1)
    for j, ln in enumerate(lines):
        text(f"{eid}_l{j}", 235, cy + 6 + j * 16, ln, font=13, color="#7c4a02")
    arrow(f"{eid}_arr", 220 + chip_w, cy + chip_h / 2, MOD_X, target_y + MOD_H / 2,
          color="#f59e0b", end_id=target_id)


# Position by module y:
m1_y = MOD_TOP                                # 100
m4_y = MOD_TOP + 3 * (MOD_H + MOD_GAP)        # 385
m6_y = MOD_TOP + 5 * (MOD_H + MOD_GAP)        # 575
m7_y = MOD_TOP + 6 * (MOD_H + MOD_GAP)        # 670

input_chip("in1", "m1", m1_y, ["input.txt", "32,033 names"])
input_chip("in4", "m4", m4_y, ["n_layer = 1", "n_embd = 16", "n_head = 4", "block_size = 16"])
input_chip("in6", "m6", m6_y, ["seed = 42", "lr = 0.01", "1000 steps"])
input_chip("in7", "m7", m7_y, ["temperature = 0.5"])


# ---------- output chips (right column) ----------

def output_chip(eid, source_id, source_y, lines, *, chip_w=320, fill=TEAL, stroke="#0e7490", text_color="#0a4f5c"):
    """Teal chip on the right, with an arrow pointing right from source module to chip."""
    chip_h = 18 + 14 * len(lines)
    cy = source_y + (MOD_H - chip_h) / 2
    arrow(f"{eid}_arr", MOD_X + MOD_W, source_y + MOD_H / 2, 820, cy + chip_h / 2,
          color=stroke, start_id=source_id)
    rect(eid, 820, cy, chip_w, chip_h, fill=fill, stroke=stroke, rounded=True, label=None, stroke_width=1)
    for j, ln in enumerate(lines):
        text(f"{eid}_l{j}", 835, cy + 6 + j * 16, ln, font=13, color=text_color)


def y_of(idx):
    return MOD_TOP + idx * (MOD_H + MOD_GAP)


output_chip("out1", "m1", y_of(0), ["docs : list[str]", "32,033 shuffled names"])
output_chip("out2", "m2", y_of(1), ["uchars (26 letters)  +  BOS (=26)", "vocab_size = 27"])
output_chip("out3", "m3", y_of(2), ["forward graph builds itself", ".backward() walks it via chain rule"])
output_chip("out4", "m4", y_of(3), ["9 weight matrices  ->  4,192 random Values", "embed 688  +  attn 1024  +  mlp 2048  +  head 432"])
output_chip("out5", "m5", y_of(4), ["one token  +  position  ->  27 logits", "(embed -> rmsnorm -> attn -> mlp -> lm_head)"])
output_chip("out6", "m6", y_of(5),
            ["loss : 3.30  ->  2.65    (~1.9x more confident)",
             "trained weights ready for inference"],
            fill="#ffe8cc", stroke="#c2410c", text_color="#7c2d12")
output_chip("out7", "m7", y_of(6),
            ["20 names sampled, BOS -> letters -> BOS"],
            fill="#e9d5ff", stroke="#7c3aed", text_color="#4c1d95")


# ---------- final generated-names callout ----------

NAMES_Y = 770
rect("names_box", 220, NAMES_Y, 920, 110, fill=GREEN, stroke="#15803d", stroke_width=2, rounded=True)
text("names_title", 240, NAMES_Y + 10, "Final output : 20 generated names  (seed = 42, T = 0.5)",
     font=16, color="#14532d")

# The 20 names from the real training run, laid out 5 per row in 4 rows.
real_names = [
    "kamon",  "ann",    "karai",  "jaire",  "vialan",
    "karia",  "yeran",  "anna",   "areli",  "kaina",
    "konna",  "keylen", "liole",  "alerin", "earan",
    "lenne",  "kana",   "lara",   "alela",  "anton",
]
COL_W = 175
for i, name in enumerate(real_names):
    col = i % 5
    row = i // 5
    text(f"name_{i}", 240 + col * COL_W, NAMES_Y + 38 + row * 18, name,
         font=14, color="#166534")


# ---------- footnote / source ----------

text("foot1", 220, 890,
     "Each numbered box maps to a section of  microgpt_original.py  (Karpathy, ~200 lines, pure Python).",
     font=12, color=SUBTLE)


# ---------- write outputs ----------

excalidraw_doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "gridSize": None,
        "viewBackgroundColor": "#ffffff",
    },
    "files": {},
}

OUT_FILE.write_text(json.dumps(excalidraw_doc, indent=2))
print(f"wrote {OUT_FILE}  ({OUT_FILE.stat().st_size:,} bytes, {len(elements)} elements)")

# Also write a minimal compact form (just the elements) for use with the MCP create_view tool
compact = [
    {"type": "cameraUpdate", "width": 1200, "height": 900, "x": 0, "y": 0}
]
# Strip the heavy default fields when emitting compact form, but keep the canonical x/y/width/height.
KEEP_KEYS = {"type", "id", "x", "y", "width", "height", "text", "fontSize",
             "strokeColor", "backgroundColor", "fillStyle", "strokeWidth",
             "strokeStyle", "roundness", "points", "endArrowhead",
             "startArrowhead", "startBinding", "endBinding", "opacity",
             "textAlign", "verticalAlign", "containerId"}
for el in elements:
    compact.append({k: v for k, v in el.items() if k in KEEP_KEYS})

COMPACT_FILE.write_text(json.dumps(compact))
print(f"wrote {COMPACT_FILE}  ({COMPACT_FILE.stat().st_size:,} bytes)")
