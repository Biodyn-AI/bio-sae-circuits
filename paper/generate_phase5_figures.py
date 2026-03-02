#!/usr/bin/env python3
"""Generate figures for Phase 5 biological knowledge extraction."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

BASE = Path("/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map")
FIG_DIR = BASE / "paper" / "figures"
P5_DIR = BASE / "experiments/phase5_knowledge_extraction"

# ─── Load Phase 5 data ───
print("Loading Phase 5 data...")
with open(P5_DIR / "step2_consensus_graph.json") as f:
    consensus = json.load(f)
with open(P5_DIR / "step4_hierarchy.json") as f:
    hierarchy = json.load(f)
with open(P5_DIR / "step5_celltype_circuits.json") as f:
    celltype = json.load(f)

# ════════════════════════════════════════════════════════════════
# Figure 11: Cross-Model Consensus & Biological Process Hierarchy
# ════════════════════════════════════════════════════════════════
print("Generating Figure 11: Cross-model consensus & process hierarchy...")
fig, axes = plt.subplots(2, 1, figsize=(10, 11))

# ── Panel A: Permutation test ──
ax = axes[0]

# Simulate permutation distribution from the stored stats
# We know: expected=107.3, observed=1142, p<0.001
# Generate approximate distribution for visualization
rng = np.random.RandomState(42)
# Regenerate the permutation test to get actual distribution
# Load the annotated edges for permutation
with open(P5_DIR / "step1_annotated_edges.json") as f:
    all_annotated = json.load(f)

GF_CONDITIONS = ["K562_K562_GF", "K562_Multi_GF", "TS_Multi_GF"]
from collections import defaultdict

def get_pairs(edges):
    pairs = set()
    for e in edges:
        if e["source_label"] != "unknown" and e["target_label"] != "unannotated":
            pairs.add((e["source_label"], e["target_label"]))
    return pairs

gf_union = set()
for cond in GF_CONDITIONS:
    gf_union.update(get_pairs(all_annotated[cond]))
scgpt_set = get_pairs(all_annotated["scGPT_TS_Multi"])

all_domains_scgpt = list(set(d for pair in scgpt_set for d in pair))
perm_counts = []
for _ in range(1000):
    shuffled = rng.permutation(all_domains_scgpt)
    domain_map = dict(zip(all_domains_scgpt, shuffled))
    shuffled_scgpt = set((domain_map.get(s, s), domain_map.get(t, t)) for s, t in scgpt_set)
    perm_counts.append(len(gf_union & shuffled_scgpt))
perm_counts = np.array(perm_counts)

ax.hist(perm_counts, bins=30, color='#90CAF9', edgecolor='#1565C0', alpha=0.8, label='Permuted (n=1,000)')
ax.axvline(x=1142, color='#C62828', linewidth=2.5, linestyle='-', label=f'Observed: 1,142')
ax.axvline(x=np.mean(perm_counts), color='#1565C0', linewidth=1.5, linestyle='--',
           label=f'Expected: {np.mean(perm_counts):.0f}')

ax.set_xlabel('Number of consensus domain pairs (GF ∩ scGPT)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('A   Cross-model consensus: 10.6× enrichment over chance (p < 0.001)', fontsize=12,
             fontweight='bold', loc='left')
ax.legend(fontsize=10, framealpha=0.9)

# Add annotation
ax.annotate(f'10.6× enrichment\np < 0.001',
            xy=(1142, 5), xytext=(800, 80),
            fontsize=10, fontweight='bold', color='#C62828',
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#C62828'))

# ── Panel B: Biological process hierarchy (layer centrality) ──
ax = axes[1]

# Get early and late domains
early = hierarchy['early_domains'][:12]
late = hierarchy['late_domains'][:12]

# Combine and sort
domains_data = []
for d in early:
    name = d['domain'].split(' (GO:')[0]  # Remove GO ID for readability
    if len(name) > 35:
        name = name[:32] + '...'
    domains_data.append((name, d['mean_layer'], d['n_occurrences'], 'early'))

for d in late:
    name = d['domain'].split(' (GO:')[0]
    if len(name) > 35:
        name = name[:32] + '...'
    # Avoid duplicates
    if name not in [x[0] for x in domains_data]:
        domains_data.append((name, d['mean_layer'], d['n_occurrences'], 'late'))

# Sort by mean layer
domains_data.sort(key=lambda x: x[1])

# Take a representative subset (first 8 early + last 8 late)
early_subset = [d for d in domains_data if d[3] == 'early'][:8]
late_subset = [d for d in domains_data if d[3] == 'late'][:8]
subset = early_subset + late_subset

names = [d[0] for d in subset]
layers = [d[1] for d in subset]
sizes = [min(max(d[2] / 20, 30), 300) for d in subset]
colors = ['#4CAF50' if d[3] == 'early' else '#E91E63' for d in subset]

y_pos = range(len(names))
ax.barh(y_pos, layers, color=colors, edgecolor='white', height=0.7, alpha=0.85)

for i, (name, layer) in enumerate(zip(names, layers)):
    ax.text(layer + 0.3, i, f'L{layer:.1f}', va='center', fontsize=8.5, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Mean layer position', fontsize=11)
ax.set_title('B   Biological process hierarchy: early signaling → late gene expression',
             fontsize=12, fontweight='bold', loc='left')
ax.set_xlim(0, 19)
ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#4CAF50', alpha=0.85, label='Early-layer (upstream)'),
                   Patch(facecolor='#E91E63', alpha=0.85, label='Late-layer (downstream)')]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

# Add arrow showing temporal flow
ax.annotate('', xy=(9, -0.8), xytext=(9, len(subset) + 0.3),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
ax.text(9.5, len(subset) // 2, 'Network\ndepth', fontsize=9, color='gray',
        ha='left', va='center', style='italic')

plt.tight_layout(h_pad=2.5)
plt.savefig(FIG_DIR / 'p4_fig11_knowledge_extraction.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved p4_fig11_knowledge_extraction.pdf")


# ════════════════════════════════════════════════════════════════
# Figure 12: Tissue-specific circuit enrichment
# ════════════════════════════════════════════════════════════════
print("Generating Figure 12: Tissue-specific enrichment...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Panel A: Tissue enrichment odds ratios ──
ax = axes[0]
enrichment = celltype['enrichment_tests']
tissues = ['immune', 'blood', 'kidney', 'universal', 'lung']
ors = [enrichment[t]['odds_ratio'] for t in tissues]
pvals = [enrichment[t]['p_value'] for t in tissues]
colors_bar = ['#C62828' if p < 0.05 else '#90CAF9' for p in pvals]

bars = ax.bar(range(len(tissues)), ors, color=colors_bar, edgecolor='white', width=0.6)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='No enrichment')
ax.set_xticks(range(len(tissues)))
ax.set_xticklabels([t.capitalize() for t in tissues], fontsize=10)
ax.set_ylabel('Odds Ratio (TS-only vs shared)', fontsize=10)
ax.set_title('A   Tissue enrichment in multi-tissue circuits', fontsize=11, fontweight='bold', loc='left')

# Add p-value annotations
for i, (or_val, p) in enumerate(zip(ors, pvals)):
    label = f'p<0.001' if p < 0.001 else f'p={p:.2f}'
    fontweight = 'bold' if p < 0.05 else 'normal'
    ax.text(i, or_val + 0.05, label, ha='center', va='bottom', fontsize=8, fontweight=fontweight)

# ── Panel B: Circuit classification ──
ax = axes[1]
ts_counts = celltype['tissue_specific_counts']
categories = ['Universal', 'Immune', 'Lung', 'Kidney', 'Blood']
counts = [
    celltype['n_universal_circuits'],
    ts_counts.get('immune', 0),
    ts_counts.get('lung', 0),
    ts_counts.get('kidney', 0),
    ts_counts.get('blood', 0),
]
colors_pie = ['#78909C', '#C62828', '#1565C0', '#4CAF50', '#FF8F00']

# Use horizontal bar instead of pie for clarity
bars = ax.barh(range(len(categories)), counts, color=colors_pie, edgecolor='white', height=0.6)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories, fontsize=10)
ax.set_xlabel('Number of circuit pairs', fontsize=10)
ax.set_title('B   Circuit classification by tissue specificity', fontsize=11, fontweight='bold', loc='left')
ax.invert_yaxis()

for i, count in enumerate(counts):
    ax.text(count + 10, i, str(count), va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG_DIR / 'p4_fig12_tissue_circuits.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved p4_fig12_tissue_circuits.pdf")

print("\nAll Phase 5 figures generated.")
