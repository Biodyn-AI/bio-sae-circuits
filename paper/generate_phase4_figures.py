#!/usr/bin/env python3
"""Generate figures for Phase 4 causal circuit tracing paper (sae_paper_v4)."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.sankey import Sankey
from pathlib import Path
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

BASE = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map")
FIG_DIR = BASE / "paper" / "figures"

# Load all analysis files
print("Loading data...")
with open(BASE / "experiments/phase1_k562/circuit_tracing/circuit_analysis.json") as f:
    k562_k562 = json.load(f)
with open(BASE / "experiments/phase3_multitissue/circuit_tracing/circuit_analysis.json") as f:
    k562_multi = json.load(f)
with open(BASE / "experiments/phase3_multitissue/circuit_tracing_ts_cells/circuit_analysis.json") as f:
    ts_multi = json.load(f)
with open(BASE / "experiments/scgpt_atlas/circuit_tracing/circuit_analysis.json") as f:
    scgpt = json.load(f)

# Color scheme
COLORS = {
    'k562_k562': '#2196F3',   # blue
    'k562_multi': '#4CAF50',  # green
    'ts_multi': '#FF9800',    # orange
    'scgpt': '#E91E63',       # pink/red
}
LABELS = {
    'k562_k562': 'GF K562/K562',
    'k562_multi': 'GF K562/Multi',
    'ts_multi': 'GF TS/Multi',
    'scgpt': 'scGPT TS/Multi',
}

# ============================================================
# FIGURE 1: Effect size comparison (bar chart with error indicators)
# ============================================================
print("Figure 1: Effect size comparison...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: Mean and median |d|
conditions = list(LABELS.values())
colors = list(COLORS.values())
means = [1.050, 0.977, 0.720, 1.396]
medians = [0.920, 0.873, 0.627, 1.187]

x = np.arange(len(conditions))
width = 0.35
bars1 = axes[0].bar(x - width/2, means, width, label='Mean |d|', color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = axes[0].bar(x + width/2, medians, width, label='Median |d|', color=colors, alpha=0.45, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel("Cohen's |d|", fontsize=11)
axes[0].set_title('A. Effect size magnitude', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['K562/\nK562', 'K562/\nMulti', 'TS/\nMulti', 'scGPT\nTS/Multi'], fontsize=9)
axes[0].legend(fontsize=9)
axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
axes[0].set_ylim(0, 1.8)

# Panel B: % strong effects
strong = [41.4, 34.4, 10.4, 65.2]
very_strong = [4.3, 2.7, 0.8, 14.0]
bars = axes[1].bar(x, strong, 0.6, color=colors, edgecolor='black', linewidth=0.5)
# Add very strong as darker overlay
axes[1].bar(x, very_strong, 0.6, color=[c for c in colors], alpha=0.4, edgecolor='black', linewidth=0.5, hatch='///')
axes[1].set_ylabel('Percentage of edges (%)', fontsize=11)
axes[1].set_title('B. Fraction of strong effects', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['K562/\nK562', 'K562/\nMulti', 'TS/\nMulti', 'scGPT\nTS/Multi'], fontsize=9)
# Add text labels
for i, (s, v) in enumerate(zip(strong, very_strong)):
    axes[1].text(i, s + 1, f'{s}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].legend([mpatches.Patch(color='gray', alpha=0.8), mpatches.Patch(color='gray', alpha=0.4, hatch='///')],
               ['|d| > 1.0', '|d| > 2.0'], fontsize=9)
axes[1].set_ylim(0, 80)

# Panel C: Inhibitory vs excitatory
inhib = [80.1, 79.9, 89.4, 65.5]
excit = [19.9, 20.1, 10.6, 34.5]
axes[2].bar(x, inhib, 0.6, label='Inhibitory', color=colors, edgecolor='black', linewidth=0.5)
axes[2].bar(x, excit, 0.6, bottom=inhib, label='Excitatory', color=colors, alpha=0.3, edgecolor='black', linewidth=0.5, hatch='\\\\\\')
for i, (inh, exc) in enumerate(zip(inhib, excit)):
    axes[2].text(i, inh/2, f'{inh}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    axes[2].text(i, inh + exc/2, f'{exc}%', ha='center', va='center', fontsize=8)
axes[2].set_ylabel('Percentage of edges (%)', fontsize=11)
axes[2].set_title('C. Inhibitory/excitatory balance', fontsize=12, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(['K562/\nK562', 'K562/\nMulti', 'TS/\nMulti', 'scGPT\nTS/Multi'], fontsize=9)
axes[2].set_ylim(0, 105)

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig1_effect_sizes.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig1_effect_sizes.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig1_effect_sizes.pdf")

# ============================================================
# FIGURE 2: Attenuation curves
# ============================================================
print("Figure 2: Attenuation curves...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: Geneformer K562/K562 attenuation
att_k562 = k562_k562['attenuation']
for src_key in ['L0', 'L5', 'L11', 'L15']:
    src_data = att_k562[src_key]
    curve = src_data['avg_attenuation_curve']
    src_layer = int(src_key[1:])
    x_layers = list(range(src_layer + 1, src_layer + 1 + len(curve)))
    axes[0].plot(x_layers, curve, 'o-', label=f'Source {src_key}', linewidth=2, markersize=4)

axes[0].set_xlabel('Downstream layer', fontsize=11)
axes[0].set_ylabel('Avg significant edges per feature', fontsize=11)
axes[0].set_title('A. Geneformer K562/K562 attenuation', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_xlim(0, 18)
axes[0].grid(True, alpha=0.3)

# Panel B: scGPT attenuation
att_scgpt = scgpt['attenuation']
for src_key in ['L0', 'L4', 'L8']:
    src_data = att_scgpt[src_key]
    curve = src_data['avg_attenuation_curve']
    src_layer = int(src_key[1:])
    x_layers = list(range(src_layer + 1, src_layer + 1 + len(curve)))
    axes[1].plot(x_layers, curve, 's-', label=f'Source {src_key}', linewidth=2, markersize=5)

axes[1].set_xlabel('Downstream layer', fontsize=11)
axes[1].set_ylabel('Avg significant edges per feature', fontsize=11)
axes[1].set_title('B. scGPT TS/Multi attenuation', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_xlim(0, 12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig2_attenuation.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig2_attenuation.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig2_attenuation.pdf")

# ============================================================
# FIGURE 3: Biological coherence comparison
# ============================================================
print("Figure 3: Biological coherence...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Shared ontology fraction
coherence = [52.9, 68.8, 68.5, 53.0]
x = np.arange(4)
bars = axes[0].bar(x, coherence, 0.6, color=colors, edgecolor='black', linewidth=0.8)
for i, v in enumerate(coherence):
    axes[0].text(i, v + 1, f'{v}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Shared ontology (%)', fontsize=11)
axes[0].set_title('A. Biological coherence of causal circuits', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['K562/\nK562', 'K562/\nMulti', 'TS/\nMulti', 'scGPT\nTS/Multi'], fontsize=9)
axes[0].set_ylim(0, 90)
# Add horizontal line at ~53%
axes[0].axhline(y=53, color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[0].text(3.5, 54, '~53% baseline', fontsize=8, color='gray', ha='right')
# Add bracket for multi-tissue (positioned above bar labels to avoid overlap)
axes[0].annotate('', xy=(1, 78), xytext=(2, 78),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
axes[0].text(1.5, 80, 'SAE-dependent\n(not cell-dependent)', ha='center', fontsize=8, fontstyle='italic')

# Panel B: Edge density per feature (layer-pair comparison)
layer_pairs = ['L0→L5', 'L0→L11', 'L0→L17', 'L5→L11', 'L5→L17', 'L11→L17']
k562_only = [200.7, 122.7, 58.5, 119.1, 61.9, 145.2]
multi_vals = [236.3, 147.2, 70.9, 140.5, 75.0, 133.6]
ratio = [m/k for m, k in zip(multi_vals, k562_only)]

x = np.arange(len(layer_pairs))
width = 0.35
axes[1].bar(x - width/2, k562_only, width, label='K562-only SAE', color=COLORS['k562_k562'], edgecolor='black', linewidth=0.5)
axes[1].bar(x + width/2, multi_vals, width, label='Multi-tissue SAE', color=COLORS['k562_multi'], edgecolor='black', linewidth=0.5)
# Add ratio labels
for i, r in enumerate(ratio):
    y_max = max(k562_only[i], multi_vals[i])
    axes[1].text(i, y_max + 5, f'{r:.2f}×', ha='center', fontsize=8, color='red' if r < 1 else 'darkgreen')
axes[1].set_ylabel('Edges per feature', fontsize=11)
axes[1].set_title('B. Multi-tissue SAEs yield denser circuits', fontsize=12, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(layer_pairs, fontsize=9, rotation=15)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig3_coherence.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig3_coherence.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig3_coherence.pdf")

# ============================================================
# FIGURE 4: Expanded interpretable biological circuits diagram
# ============================================================
print("Figure 4: Expanded biological circuit diagrams...")

NODE_FS = 8.5   # node label font size
EDGE_FS = 7.5   # edge label font size
LW = 1.6        # edge line width

# ---- Panel A: Geneformer circuits (separate figure) ----
fig_a, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(-0.5, 15)
ax.set_ylim(-1, 12.5)
ax.axis('off')

# 16 nodes across 4 layer columns
gf_nodes = [
    # L0 (x=2, blue)
    (2, 9.5, 'Nervous System\nDevelopment', 'L0', '#BBDEFB'),
    (2, 7.5, 'DNA Repair', 'L0', '#BBDEFB'),
    (2, 5.5, 'Cholesterol\nBiosynthesis', 'L0', '#BBDEFB'),
    (2, 3.5, 'MAPK Cascade', 'L0', '#BBDEFB'),
    # L1-2 (x=5.5, light green)
    (5.5, 9.5, 'Endosome\nOrganization', 'L1', '#C8E6C9'),
    (5.5, 7.5, 'DNA Damage\nResponse', 'L1', '#C8E6C9'),
    (5.5, 5.5, 'Proteasomal\nCatabolism', 'L2', '#C8E6C9'),
    # L5-6 (x=8.5, green)
    (8.5, 9.5, 'DNA Metabolic\nProcess', 'L5', '#A5D6A7'),
    (8.5, 7.5, 'Cell Cycle\nG2/M', 'L5', '#A5D6A7'),
    (8.5, 5.5, 'Centromere\nAssembly', 'L5', '#A5D6A7'),
    (8.5, 3.5, 'Kinetochore', 'L6', '#A5D6A7'),
    (8.5, 1.5, 'Protein\nCatabolism', 'L6', '#A5D6A7'),
    # L11-15 (x=12, orange/pink)
    (12, 9.5, 'Kinetochore', 'L11', '#FFCC80'),
    (12, 7.5, 'Spindle\nMicrotubules', 'L11', '#FFCC80'),
    (12, 5.5, 'Cytokinesis', 'L11', '#FFCC80'),
    (12, 3.5, 'Spindle\nCheckpoint', 'L15', '#F8BBD0'),
]

for x, y, label, layer, color in gf_nodes:
    bbox = dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor='black', linewidth=1.2)
    ax.text(x, y, f'{label}\n({layer})', ha='center', va='center', fontsize=NODE_FS,
            bbox=bbox, fontweight='bold')

# Edges with manually placed labels: (x1, y1, x2, y2, d, lx, ly)
gf_edges = [
    (2, 9.5, 5.5, 9.5, -1.32, 3.8, 10.2),
    (2, 9.5, 5.5, 5.5, -0.96, 3.2, 6.8),
    (2, 9.5, 8.5, 1.5, -1.27, 4.5, 3.8),
    (2, 7.5, 5.5, 7.5, -1.87, 3.8, 8.1),
    (2, 7.5, 8.5, 3.5, -3.47, 5.0, 6.0),
    (8.5, 9.5, 12, 9.5, -2.39, 10.2, 10.2),
    (8.5, 7.5, 12, 7.5, -1.40, 10.2, 8.1),
    (8.5, 5.5, 12, 7.5, -1.37, 10.2, 5.8),
    (12, 5.5, 12, 3.5, -0.90, 13.0, 4.5),
    (8.5, 9.5, 8.5, 1.5, -1.64, 9.6, 5.5),
]

for x1, y1, x2, y2, d, lx, ly in gf_edges:
    dx, dy = x2 - x1, y2 - y1
    rad = 0.08
    if dx == 0 and abs(dy) > 3:
        rad = 0.15
    x_start = x1 + (0.8 * np.sign(dx) if dx != 0 else 0.2)
    y_start = y1 + (-0.45 * np.sign(dy) if abs(dy) > 1 else 0)
    x_end = x2 - (0.8 * np.sign(dx) if dx != 0 else 0.2)
    y_end = y2 + (0.45 * np.sign(dy) if abs(dy) > 1 else 0)
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=LW,
                                connectionstyle=f'arc3,rad={rad}'))
    ax.text(lx, ly, f'd={d}', fontsize=EDGE_FS, color='#D32F2F', fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.1))

for lx, ll in [(2, 'Layer 0'), (5.5, 'Layers 1–2'), (8.5, 'Layers 5–6'), (12, 'Layers 11–15')]:
    ax.text(lx, 11.0, ll, ha='center', fontsize=10, fontweight='bold', color='#555')
ax.text(7, 11.8, 'Geneformer K562/K562 circuits', ha='center', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig4a_circuits_gf.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig4a_circuits_gf.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig4a_circuits_gf.pdf")

# ---- Panel B: scGPT circuits (separate figure) ----
fig_b, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(-0.5, 15)
ax.set_ylim(-1, 12.5)
ax.axis('off')

scgpt_nodes = [
    (2, 9.5, 'Protein\nCatabolism', 'L0', '#F8BBD0'),
    (2, 7.5, 'Proteasome', 'L0', '#F8BBD0'),
    (2, 5.5, 'NADH\nDehydrogenase', 'L0', '#F8BBD0'),
    (2, 3.5, 'DNA Damage\nResponse', 'L0', '#F8BBD0'),
    (5.5, 9.5, 'Chromatin\nOrganization', 'L1', '#E1BEE7'),
    (5.5, 7.5, 'Chemical Stress\nResponse', 'L2', '#E1BEE7'),
    (5.5, 5.5, 'Protein\nCatabolism', 'L3', '#E1BEE7'),
    (5.5, 3.5, 'Cell Cycle', 'L3', '#E1BEE7'),
    (8.5, 9.5, 'rRNA Cleavage\n★Hub: 6,494', 'L4', '#B2EBF2'),
    (8.5, 7.5, 'DNA\nMetabolism', 'L4', '#B2EBF2'),
    (8.5, 5.5, 'Electron\nTransport', 'L4', '#B2EBF2'),
    (8.5, 3.5, 'Macromolecule\nBiosynthesis', 'L6', '#B2EBF2'),
    (12, 8.5, 'Proteasome', 'L9', '#B2DFDB'),
    (12, 6.5, 'Protein\nCatabolism', 'L10', '#B2DFDB'),
]

for x, y, label, layer, color in scgpt_nodes:
    bbox = dict(boxstyle='round,pad=0.4', facecolor=color, edgecolor='black', linewidth=1.2)
    ax.text(x, y, f'{label}\n({layer})', ha='center', va='center', fontsize=NODE_FS,
            bbox=bbox, fontweight='bold')

scgpt_edges = [
    (2, 9.5, 5.5, 9.5, -8.19, 3.8, 10.2),
    (2, 9.5, 5.5, 7.5, -3.12, 3.0, 8.1),
    (2, 9.5, 5.5, 5.5, -3.84, 3.0, 6.5),
    (2, 9.5, 8.5, 7.5, -6.10, 5.5, 9.2),
    (8.5, 7.5, 8.5, 3.5, -3.51, 9.6, 5.5),
    (8.5, 9.5, 8.5, 3.5, -3.51, 7.5, 6.0),
    (2, 7.5, 12, 8.5, -2.50, 7.0, 8.7),
    (2, 9.5, 12, 6.5, -1.95, 7.0, 7.0),
    (2, 3.5, 5.5, 3.5, -3.51, 3.8, 4.1),
]

for x1, y1, x2, y2, d, lx, ly in scgpt_edges:
    dx, dy = x2 - x1, y2 - y1
    rad = 0.08
    if dx == 0 and abs(dy) > 3:
        rad = 0.15
    x_start = x1 + (0.8 * np.sign(dx) if dx != 0 else 0.2)
    y_start = y1 + (-0.45 * np.sign(dy) if abs(dy) > 1 else 0)
    x_end = x2 - (0.8 * np.sign(dx) if dx != 0 else 0.2)
    y_end = y2 + (0.45 * np.sign(dy) if abs(dy) > 1 else 0)
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle='->', color='#C2185B', lw=LW,
                                connectionstyle=f'arc3,rad={rad}'))
    ax.text(lx, ly, f'd={d}', fontsize=EDGE_FS, color='#C2185B', fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.1))

for lx, ll in [(2, 'Layer 0'), (5.5, 'Layers 1–3'), (8.5, 'Layers 4–6'), (12, 'Layers 9–10')]:
    ax.text(lx, 11.0, ll, ha='center', fontsize=10, fontweight='bold', color='#555')
ax.text(7, 11.8, 'scGPT TS/Multi circuits', ha='center', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig4b_circuits_scgpt.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig4b_circuits_scgpt.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig4b_circuits_scgpt.pdf")

# ============================================================
# FIGURE 5: Cross-model comparison radar/summary
# ============================================================
print("Figure 5: Cross-model comparison...")

fig, axes = plt.subplots(2, 1, figsize=(10, 11))

# Panel A: Hub feature biology comparison (horizontal bar)
ax = axes[0]
gf_hubs = [
    ('Golgi Organization', 8028),
    ('RNA Methylation', 6921),
    ('Growth Factor Resp.', 6006),
    ('Cholesterol Biosyn.', 5096),
    ('RNA Splicing', 4782),
]
scgpt_hubs = [
    ('rRNA Cleavage', 6494),
    ('Electron Transport', 6050),
    ('NADH Dehydr.', 4785),
    ('NADH Dehydr. (2)', 3849),
    ('Electron Transport (2)', 3420),
]

y_pos = np.arange(5)
ax.barh(y_pos + 0.2, [h[1] for h in gf_hubs], 0.35, color=COLORS['k562_k562'],
        label='Geneformer', edgecolor='black', linewidth=0.5)
ax.barh(y_pos - 0.2, [h[1] for h in scgpt_hubs], 0.35, color=COLORS['scgpt'],
        label='scGPT', edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ylabels = []
for g, s in zip(gf_hubs, scgpt_hubs):
    ylabels.append(f'{g[0]}\n{s[0]}')
ax.set_yticklabels(ylabels, fontsize=8)
ax.set_xlabel('Out-degree (downstream targets)', fontsize=11)
ax.set_title('A. Hub features: different organizing principles', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.invert_yaxis()

# Panel B: Normalized feature connectivity
ax = axes[1]
# Edges per feature normalized by total features
gf_edges_per = [2459, 1389, 1033, 615]  # L0, L5, L11, L15
gf_normalized = [e / 4608 * 100 for e in gf_edges_per]  # % of feature space
scgpt_edges_per = [1659, 1380, 379]  # L0, L4, L8
scgpt_normalized = [e / 2048 * 100 for e in scgpt_edges_per]

ax.plot([0, 5, 11, 15], gf_normalized, 'o-', color=COLORS['k562_k562'], linewidth=2,
        markersize=8, label='Geneformer (% of 4,608)')
ax.plot([0, 4, 8], scgpt_normalized, 's-', color=COLORS['scgpt'], linewidth=2,
        markersize=8, label='scGPT (% of 2,048)')

ax.set_xlabel('Source layer', fontsize=11)
ax.set_ylabel('% of feature space reached', fontsize=11)
ax.set_title('B. Normalized feature connectivity', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 16)

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig5_crossmodel.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig5_crossmodel.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig5_crossmodel.pdf")

# ============================================================
# FIGURE 6: Cell-type independence of coherence
# ============================================================
print("Figure 6: Cell-type independence...")

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Panel A: Same SAE, different cells
ax = axes[0]
# K562/Multi vs TS/Multi layer-pair comparison
pairs = ['L0→L5', 'L0→L11', 'L0→L17', 'L5→L11', 'L5→L17', 'L11→L17']
k562_edges = [236.3, 147.2, 70.9, 140.5, 75.0, 133.6]
ts_edges = [74.9, 39.5, 13.9, 42.8, 18.1, 52.2]

x = np.arange(len(pairs))
width = 0.35
ax.bar(x - width/2, k562_edges, width, label='K562 cells', color=COLORS['k562_multi'],
       edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, ts_edges, width, label='TS cells', color=COLORS['ts_multi'],
       edgecolor='black', linewidth=0.5)
for i in range(len(pairs)):
    ratio_val = ts_edges[i] / k562_edges[i]
    ax.text(i, max(k562_edges[i], ts_edges[i]) + 5, f'{ratio_val:.2f}×',
            ha='center', fontsize=7.5, color='red')
ax.set_ylabel('Edges per feature', fontsize=11)
ax.set_title('A. Circuit density: cell-type dependent', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pairs, fontsize=9, rotation=15)
ax.legend(fontsize=9)

# Panel B: Coherence is SAE-dependent
ax = axes[1]
# Group by SAE type
sae_types = ['K562-only\nSAE', 'Multi-tissue\nSAE']
k562_sae_coh = [52.9]  # only K562/K562
multi_sae_coh = [68.8, 68.5]  # K562/Multi, TS/Multi

# Bar plot showing coherence
x_k562 = [0]
x_multi = [1, 1.6]
ax.bar(x_k562, k562_sae_coh, 0.5, color=COLORS['k562_k562'], edgecolor='black', linewidth=0.8,
       label='K562 cells')
ax.bar([1], [multi_sae_coh[0]], 0.5, color=COLORS['k562_multi'], edgecolor='black', linewidth=0.8,
       label='K562 cells (multi SAE)')
ax.bar([1.6], [multi_sae_coh[1]], 0.5, color=COLORS['ts_multi'], edgecolor='black', linewidth=0.8,
       label='TS cells (multi SAE)')

for xi, v in zip([0, 1, 1.6], [52.9, 68.8, 68.5]):
    ax.text(xi, v + 1, f'{v}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add bracket (positioned well above bar labels to avoid overlap)
ax.annotate('', xy=(0.75, 78), xytext=(1.85, 78),
            arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
ax.text(1.3, 80, 'Δ = 0.3%\n(not significant)', ha='center', fontsize=9, color='darkgreen',
        fontstyle='italic')

ax.set_ylabel('Shared ontology (%)', fontsize=11)
ax.set_title('B. Coherence: SAE-dependent, not cell-dependent', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1.3])
ax.set_xticklabels(sae_types, fontsize=10)
ax.set_ylim(0, 92)
ax.axhline(y=53, color='gray', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig6_celltype.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig6_celltype.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig6_celltype.pdf")

# ============================================================
# FIGURE 7: Biological pathway flow across layers
# ============================================================
print("Figure 7: Multi-tissue cascade...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(-0.5, 18.5)
ax.set_ylim(-1.2, 5.5)
ax.axis('off')
ax.set_title('DNA Damage → Cell Cycle Arrest cascade (Geneformer multi-tissue SAE)',
             fontsize=13, fontweight='bold', pad=15)

# DNA damage response cascade through layers
cascade_nodes = [
    (1, 3.5, 'DNA Damage\nResponse\n(L0_F2551)', '#FFCDD2', 'L0'),
    (5, 3.5, 'DNA Damage\nResponse\n(L5_F3538)', '#EF9A9A', 'L5'),
    (5, 1.5, 'Centromere\nAssembly\n(L5_F3098)', '#CE93D8', 'L5'),
    (9, 3.5, 'G2/M\nTransition\n(L11_F3296)', '#A5D6A7', 'L11'),
    (13, 3.5, 'G2/M\nTransition\n(L17_F1269)', '#81C784', 'L17'),
    (13, 1.5, 'G2/M\nTransition\n(L17_F610)', '#81C784', 'L17'),
]

for x, y, label, color, layer in cascade_nodes:
    bbox = dict(boxstyle='round,pad=0.5', facecolor=color, edgecolor='black', linewidth=1.5)
    ax.text(x, y, label, ha='center', va='center', fontsize=8.5, bbox=bbox, fontweight='bold')

# Edges with manually positioned labels to avoid overlap
# (x1, y1, x2, y2, d, bio_label, label_x, label_y)
cascade_edges = [
    (1, 3.5, 5, 3.5, -3.84, 'Mitotic regulation',       3.0, 4.5),
    (5, 3.5, 9, 3.5, -1.57, 'Chromatid segregation',     7.0, 4.5),
    (5, 3.5, 13, 3.5, -2.66, 'p53 / sister chromatid',   10.5, 4.5),
    (5, 1.5, 9, 3.5, -1.37, 'Nuclear div. / kinetochore', 6.3, 0.5),
    (5, 1.5, 13, 3.5, -1.76, 'APC/C / spindle ckpt.',    9.5, -0.2),
    (9, 3.5, 13, 1.5, -0.85, 'Cell division',             11.5, 2.8),
    (1, 3.5, 13, 3.5, -2.30, 'Long-range DDR→G2/M',      3.0, -0.7),
]

for x1, y1, x2, y2, d, bio, lx, ly in cascade_edges:
    rad = 0.15 if abs(y2-y1) > 0.5 else 0.05
    # Use larger rad for the long-range bottom edge
    if x2 - x1 > 10:
        rad = -0.25  # curve below
    ax.annotate('', xy=(x2 - 0.9, y2), xytext=(x1 + 0.9, y1),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=2.0,
                                connectionstyle=f'arc3,rad={rad}'))
    ax.text(lx, ly, f'd = {d}  {bio}', fontsize=7, ha='center', va='center',
            color='#B71C1C', fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='#E0E0E0', alpha=0.9, pad=0.2, linewidth=0.5))

# Layer markers
for lx, ll in [(1, 'L0'), (5, 'L5'), (9, 'L11'), (13, 'L17')]:
    ax.text(lx, 5.0, ll, ha='center', fontsize=11, fontweight='bold', color='#37474F')

# Legend annotation
ax.text(16.5, 3.5, 'Biological\nprogression:', fontsize=9, fontweight='bold', color='#333')
ax.text(16.5, 2.4, 'DNA damage\ndetection →\nchromatin repair →\ncell cycle arrest', fontsize=8, color='#555',
        fontstyle='italic')

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig7_cascade.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig7_cascade.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig7_cascade.pdf")

# ============================================================
# FIGURE 8: scGPT Proteostasis / Stress Response Cascade
# ============================================================
print("Figure 8: scGPT stress response cascade...")

fig, ax = plt.subplots(1, 1, figsize=(15, 6.5))
ax.set_xlim(-0.5, 16)
ax.set_ylim(-1.5, 5.5)
ax.axis('off')
ax.set_title('scGPT: Protein Quality Control → Stress Response → Biosynthetic Recovery',
             fontsize=13, fontweight='bold', pad=15)

# Nodes organized by layer and biological role
nodes_8 = [
    # Source hub (L0)
    (1, 3.5, 'Protein\nCatabolism\n(L0_F507)', '#F8BBD0', 'L0'),
    (1, 1.0, 'Proteasome\n(L0_F379)', '#F8BBD0', 'L0'),
    # Early response (L1-2)
    (4.5, 4.5, 'Chromatin\nOrganization\n(L1)', '#E1BEE7', 'L1'),
    (4.5, 2.5, 'Chemical Stress\nResponse\n(L2)', '#E1BEE7', 'L2'),
    # Mid processing (L3-4)
    (8, 4.5, 'Protein\nCatabolism\n(L3)', '#CE93D8', 'L3'),
    (8, 2.5, 'DNA\nMetabolism\n(L4)', '#BA68C8', 'L4'),
    (8, 0.5, 'rRNA Cleavage\n★Hub\n(L4_F446)', '#BA68C8', 'L4'),
    # Recovery (L6-10)
    (11.5, 4.5, 'Macromolecule\nBiosynthesis\n(L6)', '#B2EBF2', 'L6'),
    (11.5, 2.5, 'Proteasome\n(L9)', '#80DEEA', 'L9'),
    (11.5, 0.5, 'Protein\nCatabolism\n(L10)', '#80DEEA', 'L10'),
]

for x, y, label, color, layer in nodes_8:
    bbox = dict(boxstyle='round,pad=0.45', facecolor=color, edgecolor='black', linewidth=1.3)
    ax.text(x, y, label, ha='center', va='center', fontsize=7.5, bbox=bbox, fontweight='bold')

# Edges with manual label positions
edges_8 = [
    # From Protein Catabolism hub
    (1, 3.5, 4.5, 4.5, -8.19, 'β-catenin degradation\n161 shared terms', 2.8, 4.8),
    (1, 3.5, 4.5, 2.5, -3.12, 'Chemical stress detection\n152 shared terms', 2.8, 2.2),
    (1, 3.5, 8, 4.5, -3.84, 'Apoptotic cascade\n153 shared terms', 5.0, 4.0),
    (1, 3.5, 8, 2.5, -6.10, 'ER-phagosome pathway\n158 shared terms', 5.0, 1.5),
    (1, 3.5, 11.5, 0.5, -1.95, 'Long-range protein QC\n153 shared terms', 6.5, -0.5),
    # Mid → Late
    (8, 2.5, 11.5, 4.5, -3.51, 'Biosynthetic\nrecovery', 10.0, 4.0),
    (8, 0.5, 11.5, 4.5, -3.51, 'Ribosome →\nbiosynthesis', 10.0, 1.5),
    # Proteasome persistence
    (1, 1.0, 11.5, 2.5, -2.50, 'Cross-layer proteostasis\n141 shared terms', 6.5, 0.5),
]

for x1, y1, x2, y2, d, bio, lx, ly in edges_8:
    rad = 0.1 if abs(y2-y1) < 1 else 0.12
    if x2 - x1 > 8:
        rad = -0.15  # curve below for long-range edges
    ax.annotate('', xy=(x2 - 1.0, y2), xytext=(x1 + 1.0, y1),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.8,
                                connectionstyle=f'arc3,rad={rad}'))
    ax.text(lx, ly, f'd = {d}  {bio}', fontsize=6.5, ha='center', va='center',
            color='#B71C1C', fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='#E0E0E0', alpha=0.92, pad=0.2, linewidth=0.5))

# Layer markers
for lx, ll in [(1, 'L0'), (4.5, 'L1–2'), (8, 'L3–4'), (11.5, 'L6–10')]:
    ax.text(lx, 5.2, ll, ha='center', fontsize=11, fontweight='bold', color='#37474F')

# Biological flow annotation
ax.annotate('', xy=(13.5, 3.5), xytext=(13.5, 1.5),
            arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=2.5))
ax.text(14.5, 3.5, 'Protein\nquality\ncontrol', fontsize=8, fontweight='bold', color='#1B5E20', va='top')
ax.text(14.5, 2.5, '↓ stress\ndetection', fontsize=7, color='#2E7D32', va='top')
ax.text(14.5, 1.5, '↓ repair\n& recovery', fontsize=7, color='#388E3C', va='top')

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig8_scgpt_cascade.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig8_scgpt_cascade.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig8_scgpt_cascade.pdf")


# ============================================================
# FIGURE 9: Geneformer Neurodevelopment-Proteostasis Hub
# ============================================================
print("Figure 9: Neurodevelopment-proteostasis hub...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(-0.5, 16)
ax.set_ylim(-0.5, 6)
ax.axis('off')
ax.set_title('Geneformer: Nervous System Development Hub → Proteostasis Targets',
             fontsize=13, fontweight='bold', pad=15)

# Central hub at left, targets fan out across layers
nodes_9 = [
    # Hub (L0)
    (1.5, 3.0, 'Nervous System\nDevelopment\n(L0_F146)\n★Top hub', '#BBDEFB', 'L0'),
    # Immediate targets (L1)
    (5, 5.0, 'Endosome\nOrganization\n(L1)', '#C8E6C9', 'L1'),
    (5, 3.0, 'DNA Damage\nResponse\n(L1)', '#FFCDD2', 'L1'),
    # L2 targets
    (8, 5.0, 'Proteasomal\nCatabolism\n(L2)', '#E1BEE7', 'L2'),
    (8, 3.0, 'Ribosome\nBiogenesis\n(L2)', '#E1BEE7', 'L2'),
    # L6 targets
    (11, 5.0, 'Protein\nCatabolism\n(L6)', '#FFCC80', 'L6'),
    (11, 3.0, 'NF-κB\nSignaling\n(L6)', '#FFCC80', 'L6'),
    (11, 1.0, 'Modification-Dep\nProtein Catab\n(L6)', '#FFCC80', 'L6'),
    # L13 target
    (14, 3.0, 'Golgi Vesicle\nTransport\n(L13)', '#B2DFDB', 'L13'),
]

for x, y, label, color, layer in nodes_9:
    lw = 2.0 if 'Top hub' in label else 1.3
    bbox = dict(boxstyle='round,pad=0.45', facecolor=color, edgecolor='black', linewidth=lw)
    ax.text(x, y, label, ha='center', va='center', fontsize=7.5, bbox=bbox, fontweight='bold')

# Edges from hub to targets
edges_9 = [
    (1.5, 3.0, 5, 5.0, -1.32, '142 shared terms\nNeurodegeneration, lysosomal', 3.3, 4.7),
    (1.5, 3.0, 5, 3.0, -1.16, '139 shared terms\nDNA repair', 3.3, 2.2),
    (1.5, 3.0, 8, 5.0, -0.96, '140 shared terms\nUbiquitin-proteasome', 5.0, 5.5),
    (1.5, 3.0, 8, 3.0, -0.88, '128 shared terms\nrRNA processing', 5.0, 1.5),
    (1.5, 3.0, 11, 5.0, -1.27, '139 shared terms\nNF-κB, immune response', 6.5, 4.3),
    (1.5, 3.0, 11, 1.0, -1.05, '136 shared terms\nApoptosis', 6.5, 0.5),
    (1.5, 3.0, 14, 3.0, -0.81, '141 shared terms\nSecretory pathway', 8.0, 2.2),
]

for x1, y1, x2, y2, d, bio, lx, ly in edges_9:
    rad = 0.08 if abs(y2 - y1) < 1 else 0.12
    ax.annotate('', xy=(x2 - 1.1, y2), xytext=(x1 + 1.1, y1),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.8,
                                connectionstyle=f'arc3,rad={rad}'))
    ax.text(lx, ly, f'd = {d}  {bio}', fontsize=6.5, ha='center', va='center',
            color='#0D47A1', fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='#BBDEFB', alpha=0.92, pad=0.2, linewidth=0.5))

# Layer markers
for lx, ll in [(1.5, 'L0'), (5, 'L1'), (8, 'L2'), (11, 'L6'), (14, 'L13')]:
    ax.text(lx, 5.8, ll, ha='center', fontsize=10, fontweight='bold', color='#37474F')

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig9_neuro_hub.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig9_neuro_hub.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig9_neuro_hub.pdf")


# ============================================================
# FIGURE 10: Geneformer Mitotic Progression Circuit
# ============================================================
print("Figure 10: Mitotic progression circuit...")

fig, ax = plt.subplots(1, 1, figsize=(15, 6))
ax.set_xlim(-0.5, 17)
ax.set_ylim(-1.0, 5.5)
ax.axis('off')
ax.set_title('Geneformer: Mitotic Progression Circuit (DNA Replication → Spindle → Cytokinesis)',
             fontsize=13, fontweight='bold', pad=15)

nodes_10 = [
    # L0 features
    (1, 4.0, 'DNA Repair\n(L0_F3717)', '#BBDEFB', 'L0'),
    (1, 1.5, 'Cholesterol\nBiosynthesis\n(L0_F3402)\n★Hub: 5,096', '#BBDEFB', 'L0'),
    # L5 features
    (5, 4.0, 'DNA Metabolic\nProcess\n(L5_F3780)', '#A5D6A7', 'L5'),
    (5, 2.0, 'Cell Cycle\nG2/M\n(L5_F3300)', '#A5D6A7', 'L5'),
    (5, 0.0, 'Centromere\nAssembly\n(L5_F3098)', '#CE93D8', 'L5'),
    # L6-7 features
    (8.5, 4.0, 'Kinetochore\n(L6_F2814)', '#FFCC80', 'L6'),
    (8.5, 2.0, 'G2/M\nTransition\n(L7)', '#FFCC80', 'L7'),
    # L11 features
    (12, 4.0, 'Kinetochore\n(L11_F3000)', '#FF8A65', 'L11'),
    (12, 2.0, 'Spindle\nMicrotubules\n(L11)', '#FF8A65', 'L11'),
    (12, 0.0, 'Cytokinesis\n(L11)', '#FF8A65', 'L11'),
    # L15 feature
    (15, 1.0, 'Spindle\nCheckpoint\n(L15)', '#F8BBD0', 'L15'),
]

for x, y, label, color, layer in nodes_10:
    lw = 2.0 if 'Hub' in label else 1.3
    bbox = dict(boxstyle='round,pad=0.45', facecolor=color, edgecolor='black', linewidth=lw)
    ax.text(x, y, label, ha='center', va='center', fontsize=7.5, bbox=bbox, fontweight='bold')

edges_10 = [
    # DNA repair → kinetochore (long-range)
    (1, 4.0, 8.5, 4.0, -3.47, 'DNA damage → mitotic checkpoint', 4.8, 4.7),
    # DNA metabolic → downstream
    (5, 4.0, 8.5, 2.0, -2.45, 'Replication → division', 6.8, 3.6),
    (5, 4.0, 12, 4.0, -2.39, 'Replication → spindle', 8.5, 4.7),
    # Cell cycle → spindle
    (5, 2.0, 12, 2.0, -1.40, 'G2/M commitment → assembly', 8.5, 2.5),
    # Centromere → spindle
    (5, 0.0, 12, 2.0, -1.37, 'Centromere → kinetochore', 8.5, 0.3),
    # Centromere → G2/M transition
    (5, 0.0, 8.5, 2.0, -1.37, 'Nuclear div.', 6.3, 0.5),
    # Cytokinesis → checkpoint
    (12, 0.0, 15, 1.0, -0.90, 'Late mitotic feedback', 13.5, -0.3),
    # Kinetochore(L6) → Kinetochore(L11)
    (8.5, 4.0, 12, 4.0, -1.89, 'Cross-layer persistence', 10.3, 3.4),
]

for x1, y1, x2, y2, d, bio, lx, ly in edges_10:
    rad = 0.08 if abs(y2 - y1) < 1.5 else 0.12
    ax.annotate('', xy=(x2 - 1.0, y2), xytext=(x1 + 1.0, y1),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.8,
                                connectionstyle=f'arc3,rad={rad}'))
    ax.text(lx, ly, f'd = {d}  {bio}', fontsize=6.5, ha='center', va='center',
            color='#B71C1C', fontstyle='italic',
            bbox=dict(facecolor='white', edgecolor='#E0E0E0', alpha=0.92, pad=0.2, linewidth=0.5))

# Layer markers
for lx, ll in [(1, 'L0'), (5, 'L5'), (8.5, 'L6–7'), (12, 'L11'), (15, 'L15')]:
    ax.text(lx, 5.2, ll, ha='center', fontsize=10, fontweight='bold', color='#37474F')

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig10_mitotic.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'p4_fig10_mitotic.png', dpi=300, bbox_inches='tight')
print("  Saved p4_fig10_mitotic.pdf")


print("\nAll figures generated successfully!")
