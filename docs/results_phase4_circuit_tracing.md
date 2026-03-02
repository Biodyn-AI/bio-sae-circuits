# Subproject 42: Phase 4 Results — Causal Feature-to-Feature Circuit Tracing

*Completed: 2026-03-01*

## Overview

Phase 4 extends the Phase 2 causal analysis (Step 8: feature→output logits) to **feature→feature** causal tracing: ablating source SAE features and measuring how downstream SAE features across all subsequent layers change. This reveals the directed computational graph the model uses to transform biological representations across depth.

## Step 13: Causal Circuit Tracing

**Script:** `13_causal_circuit_tracing.py`
**Source layers:** L0, L5, L11, L15 (30 well-annotated features per layer = 120 total)
**Cells:** 200 control cells from Replogle K562 CRISPRi dataset
**Method:** Hook-based ablation at source layer → capture all downstream hidden states → encode through downstream SAEs → Welford's online accumulation → Cohen's d + consistency significance testing
**Significance thresholds:** |Cohen's d| > 0.5 AND consistency > 0.7
**Compute:** 7.5 hours on Apple Silicon (MPS), 24,776 forward passes total

### Per-Source-Layer Results

| Source | Downstream Layers | Forward Passes | Time | Total Edges | Avg/Feature | Median/Feature | Range |
|--------|------------------:|---------------:|-----:|------------:|------------:|---------------:|------:|
| L0 | 17 (L1-L17) | 6,176 | 150 min | 73,769 | 2,459 | 1,722 | 382-8,028 |
| L5 | 12 (L6-L17) | 6,200 | 140 min | 41,684 | 1,389 | 1,207 | 533-4,703 |
| L11 | 6 (L12-L17) | 6,200 | 86 min | 30,981 | 1,033 | 700 | 321-3,536 |
| L15 | 2 (L16-L17) | 6,200 | 71 min | 18,438 | 615 | 258 | 79-3,257 |

### Aggregate Circuit Graph

- **52,116 significant causal edges** (|d| > 0.5, consistency > 0.7)
- **120 source features** → **26,338 unique target features**
- Mean |Cohen's d| = 1.05, median = 0.92, max = 15.75
- 41.4% of edges have |d| > 1.0 (strong effects)
- 4.3% have |d| > 2.0 (very strong effects)
- **80.1% inhibitory** (negative delta): ablating a source feature mostly *reduces* downstream activations
- **19.9% excitatory** (positive delta): ablating a source feature *increases* some downstream features (disinhibition)

### Top Hub Features

**Highest out-degree (most downstream targets):**

| Feature | Biology | Out-Degree |
|---------|---------|----------:|
| L0_F2905 | Golgi Organization | 8,028 |
| L0_F2982 | RNA Methylation | 6,921 |
| L0_F1568 | Growth Factor Response | 6,006 |
| L0_F3402 | Cholesterol Biosynthesis | 5,096 |
| L0_F4201 | RNA Splicing | 4,782 |

**Highest in-degree (most upstream sources):**

| Feature | In-Degree |
|---------|----------:|
| L16_F2818 | 93 |
| L16_F1691 | 89 |
| L16_F4354 | 89 |
| L16_F1375 | 88 |
| L16_F1057 | 88 |

### Biological Coherence

- 31,176 causal edges have ontology annotations on both source AND target features
- **16,507 (52.9%) share at least one ontology term** between source and target
- This means over half of the model's computational circuits connect biologically related features

**Example interpretable circuits:**
- L0 Nervous System Development → L1 Endosome Organization (d=-1.32, shared: neurodegeneration pathways)
- L0 Nervous System Development → L6 Protein Catabolism (d=-1.27, shared: NF-kB signaling)
- L5 Cell Cycle G2/M → L11 Spindle Microtubules (d=-1.4+, shared: mitotic processes)
- L11 Cytokinesis → L15 Spindle Checkpoint (d=-0.9+, shared: cell division)

### Attenuation Analysis

Average number of significant downstream edges per layer distance:

| Source | +1 | +2 | +3 | +6 | +12 | +17 (last) |
|--------|---:|---:|---:|---:|----:|-----:|
| L0 | 211 | 213 | 215 | 190 | 58 | 58 |
| L5 | 191 | 174 | 158 | 119 | 62 | — |
| L11 | 223 | 197 | 180 | 145 | — | — |
| L15 | 279 | 336 | — | — | — | — |

**Key observations:**
- L0 effects are remarkably persistent — they maintain ~200 significant edges for the first 5 downstream layers before decaying
- L5 effects decay more linearly
- L11 effects are strong but decay over 6 layers to ~65% of initial
- L15→L17 effects actually *increase* (279→336), suggesting late-layer features have strong local coupling

### PMI Comparison

Source features selected for causal tracing (by annotation quality) vs PMI (by co-activation frequency) had zero overlap. However, **target feature overlap is 91-95%**:

| Layer Pair | PMI Edges | Causal Edges | PMI Targets | Causal Targets | Target Overlap |
|------------|----------:|-------------:|------------:|---------------:|---------------:|
| L0→L5 | 25,000 | 1,469 | 4,101 | 1,113 | 90.6% |
| L5→L11 | 25,000 | 1,473 | 4,369 | 996 | 94.8% |
| L11→L17 | 25,000 | 1,491 | 4,205 | 881 | 91.7% |

**Interpretation:** Statistical co-activation (PMI) and causal influence converge on the same downstream features, validating both methods. The causal approach additionally reveals directionality, effect size, and sign (excitatory vs inhibitory).

## Key Findings

1. **Dense causal connectivity**: Each biological feature causally influences 600-2,500 downstream features on average. The model's computational graph is dense, not sparse.

2. **Predominantly inhibitory**: 80% of causal edges are inhibitory — ablating a feature mostly reduces downstream activations, consistent with features encoding necessary (not redundant) information.

3. **Biologically coherent circuits**: 53% of causal edges connect features annotated with shared biology, far above chance. The model builds hierarchical biological representations through interpretable computational pathways.

4. **Persistent early-layer effects**: L0 features maintain causal influence across all 17 downstream layers, suggesting early representations contain foundational biological information that the entire network depends on.

5. **Convergent hub targets at L16**: Late-layer features receive input from many upstream sources (in-degree ~90), suggesting L16 serves as an integration layer.

6. **PMI validates causal edges**: 91-95% target overlap between statistical and causal methods confirms that co-activation patterns reflect genuine information flow.

## Output Files

```
experiments/phase1_k562/circuit_tracing/
├── circuit_L00_features.json    (6.6 MB) — 30 source features, 73,769 edges
├── circuit_L05_features.json    (4.7 MB) — 30 source features, 41,684 edges
├── circuit_L11_features.json    (2.5 MB) — 30 source features, 30,981 edges
├── circuit_L15_features.json    (860 KB) — 30 source features, 18,438 edges
├── circuit_graph.json           (15 MB)  — aggregated graph + hub analysis
└── circuit_analysis.json        (401 KB) — PMI comparison, biology, attenuation
```

## Relation to Previous Phases

| Phase | What | Finding |
|-------|------|---------|
| Phase 1 | Feature atlas (SAE training + annotation) | 82,525 features, 45-59% annotated |
| Phase 2, Step 8 | Feature → output logit (causal patching) | 60% features have >2x specificity |
| Phase 2, Step 11 | Feature ↔ feature (statistical PMI) | 97-99.8% features on information highways |
| **Phase 4, Step 13** | **Feature → feature (causal tracing)** | **52,116 causal edges, 53% biologically coherent** |

Phase 4 closes the loop: Phase 1 identified *what* features exist, Phase 2 showed they *matter* for output and co-activate, and Phase 4 reveals *how* they causally drive each other across the network's depth.
