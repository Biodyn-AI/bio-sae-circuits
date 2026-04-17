import os
"""
F1 — Pipeline schematic for causal circuit tracing.

Clean horizontal two-row layout designed for the Bioinformatics two-column
figure* environment. No overlaps, readable text, explicit arrows.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

OUT_DIR = Path(os.environ.get("FIG_OUT_DIR", "../../paper/figures"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(15.0, 5.2))
ax.set_xlim(0, 30)
ax.set_ylim(0, 10.5)
ax.set_aspect("equal")
ax.axis("off")

# Palette
C_INPUT = "#E3F2FD"
C_MODEL = "#B3E5FC"
C_SAE = "#FFF9C4"
C_ABL = "#FFCDD2"
C_METRIC = "#D1C4E9"
C_PRIOR = "#C8E6C9"
ARROW = "#37474F"

FS_BODY = 10
FS_TITLE = 10.5


def box(x, y, w, h, txt, color, title=None):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor=color, edgecolor="black", linewidth=1.0, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, txt,
        ha="center", va="center",
        fontsize=FS_BODY, zorder=3, linespacing=1.2,
    )
    if title:
        ax.text(
            x + w / 2, y + h + 0.25, title,
            ha="center", va="bottom",
            fontsize=FS_TITLE, fontweight="bold", color="#263238", zorder=3,
        )


def arrow(x1, y1, x2, y2, style="->", lw=1.4, rad=0.0, color=ARROW):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        mutation_scale=16,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(a)


# ---------------- Top row: 5 pipeline stages (y = 6.3 .. 8.7) ----------------
TOP_Y = 6.3
BOX_H = 2.4
# column layout: box width 5.0, gap 0.7, starting x = 0.3
X0, W, GAP = 0.3, 5.0, 0.7

stages = [
    (C_INPUT,
     "200 K562 control\ncells (Replogle)\n— or 200 TS cells\n(immune/kidney/lung)",
     "1. Input cells"),
    (C_INPUT,
     "Rank-value tokens\n(Geneformer)\n— or HVG + values\n(scGPT, padded 1200)",
     "2. Tokenization"),
    (C_MODEL,
     "Geneformer V2-316M\n(18 layers, d=1152)\n— or scGPT\n(12 layers, d=512)",
     "3. Foundation model"),
    (C_SAE,
     "SAE-encode at $\\ell_{src}$;\nzero source feature $f_i$;\ndecode; replace\nresidual stream",
     "4. Ablation"),
    (C_ABL,
     "Ablated forward pass;\nSAE-encode all\n$\\ell > \\ell_{src}$;\nper-cell deltas",
     "5. Downstream encode"),
]

for i, (c, body, title) in enumerate(stages):
    x = X0 + i * (W + GAP)
    box(x, TOP_Y, W, BOX_H, body, c, title=title)

# Connecting arrows on top row
for i in range(len(stages) - 1):
    xL = X0 + i * (W + GAP) + W
    xR = X0 + (i + 1) * (W + GAP)
    arrow(xL, TOP_Y + BOX_H / 2, xR, TOP_Y + BOX_H / 2)

# Arrow from last top-row box wrapping down to bottom row start
xlast = X0 + (len(stages) - 1) * (W + GAP) + W / 2
arrow(xlast, TOP_Y, xlast, 5.0, rad=0.0)

# ---------------- Bottom row (y = 2.6 .. 5.0) ----------------
BOT_Y = 2.6

bottom_stages_right_to_left = [
    (C_METRIC,
     "Per-cell $\\Delta$ aggregated\n(Welford) → Cohen's $d$\n+ consistency",
     "6. Edge statistic"),
    (C_METRIC,
     "Filter $|d|>0.5$,\nconsistency $>0.7$\n→ 96,892 directed\nedges (4 conditions)",
     "7. Graph assembly"),
    (C_PRIOR,
     "Ontology nulls · ChIP-seq\nK562 · CRISPRi K562 +\nRPE1 · Shifrut T cells ·\nmarginal co-expression",
     "8. Priors + validation"),
]

for i, (c, body, title) in enumerate(bottom_stages_right_to_left):
    # reverse order: right to left
    idx_from_right = i
    x = X0 + (len(stages) - 1 - idx_from_right) * (W + GAP)
    box(x, BOT_Y, W, BOX_H, body, c, title=title)

# Arrows: right box ← middle ← left on bottom row
positions = [X0 + (len(stages) - 1 - i) * (W + GAP) for i in range(len(bottom_stages_right_to_left))]
for i in range(len(positions) - 1):
    xR = positions[i]                  # right edge of right box
    xL = positions[i + 1] + W          # left edge of left (next) box
    arrow(xR, BOT_Y + BOX_H / 2, xL, BOT_Y + BOX_H / 2)

# Legend at the very bottom
legend_handles = [
    mpatches.Patch(facecolor=C_INPUT, edgecolor="black", label="Input / tokens"),
    mpatches.Patch(facecolor=C_MODEL, edgecolor="black", label="Foundation model"),
    mpatches.Patch(facecolor=C_SAE, edgecolor="black", label="SAE encode"),
    mpatches.Patch(facecolor=C_ABL, edgecolor="black", label="Ablated forward pass"),
    mpatches.Patch(facecolor=C_METRIC, edgecolor="black", label="Edge / graph"),
    mpatches.Patch(facecolor=C_PRIOR, edgecolor="black", label="Priors + validation"),
]
leg = ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.12),
    ncol=6, fontsize=10, frameon=True, edgecolor="#90A4AE",
)

ax.set_title(
    "Causal circuit tracing pipeline",
    fontsize=14, fontweight="bold", y=1.02,
)

plt.tight_layout()
pdf = OUT_DIR / "p4_fig_pipeline.pdf"
png = OUT_DIR / "p4_fig_pipeline.png"
plt.savefig(pdf, dpi=300, bbox_inches="tight")
plt.savefig(png, dpi=200, bbox_inches="tight")
print(f"Saved {pdf}")
print(f"Saved {png}")
