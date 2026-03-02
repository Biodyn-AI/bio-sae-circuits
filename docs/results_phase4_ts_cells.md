# Subproject 42: Phase 4 Tabula Sapiens Cells — Causal Circuit Tracing

*Completed: 2026-03-01*

## Overview

Runs causal circuit tracing on **Tabula Sapiens cells** (immune + kidney + lung tissues) instead of K562 leukemia cells, using the multi-tissue SAEs from Phase 3. This reveals whether the model's computational circuits are cell-type-specific or universal.

## Configuration

**Script:** `13_causal_circuit_tracing.py` with `--data-source tabula_sapiens`
**Cells:** 200 Tabula Sapiens cells (67 immune/41 cell types + 67 kidney/13 cell types + 66 lung/34 cell types), stratified sampling
**SAE models:** Multi-tissue SAEs at layers 0, 5, 11, 17
**Source layers:** L0, L5, L11 (downstream tracing restricted to available SAE layers)
**Compute:** 3.2 hours on Apple Silicon (MPS)

## Three-Way Comparison

We now have three circuit tracing runs for comparison:

1. **K562/K562**: K562 cells + K562-only SAEs (Phase 4 original, all 18 layers)
2. **K562/Multi**: K562 cells + Multi-tissue SAEs (Phase 4 multi-tissue, 4 layers)
3. **TS/Multi**: Tabula Sapiens cells + Multi-tissue SAEs (this run, 4 layers)

### Aggregate Statistics

| Metric | K562/K562 | K562/Multi | TS/Multi |
|--------|----------:|-----------:|---------:|
| Total edges | 52,116 | 8,298 | 5,098 |
| Source features | 120 | 90 | 90 |
| Target features | 26,338 | 4,171 | 2,962 |
| Mean |Cohen's d| | 1.05 | 0.98 | 0.72 |
| Median |d| | 0.92 | 0.87 | 0.63 |
| |d| > 1.0 (strong) | 41.4% | 34.4% | 10.4% |
| **Inhibitory %** | **80.1%** | **79.9%** | **89.4%** |
| Annotated pairs | 31,176 | 4,591 | 2,805 |
| **Shared ontology %** | **52.9%** | **68.8%** | **68.5%** |

### Per-Layer-Pair Edges per Feature

| Layer Pair | K562/K562 | K562/Multi | TS/Multi | TS/K562 ratio |
|------------|----------:|-----------:|---------:|--------------:|
| L0→L5 | 200.7 | 236.3 | 74.9 | 0.32 |
| L0→L11 | 122.7 | 147.2 | 39.5 | 0.27 |
| L0→L17 | 58.5 | 70.9 | 13.9 | 0.20 |
| L5→L11 | 119.1 | 140.5 | 42.8 | 0.30 |
| L5→L17 | 61.9 | 75.0 | 18.1 | 0.24 |
| L11→L17 | 145.2 | 133.6 | 52.2 | 0.39 |

### Attenuation (avg significant edges per downstream layer)

| Source | K562/K562 | K562/Multi | TS/Multi |
|--------|----------:|-----------:|---------:|
| L0 (to L5) | 211 | 236 | 75 |
| L0 (to L11) | 213 | 147 | 39 |
| L0 (to L17) | 215 | 71 | 14 |
| L5 (to L11) | 191 | 140 | 43 |
| L5 (to L17) | 174 | 75 | 18 |
| L11 (to L17) | 223 | 134 | 52 |

## Key Findings

### 1. Tabula Sapiens circuits are dramatically sparser

TS cells produce **3-5x fewer significant causal edges** per feature compared to K562 cells with the same SAEs (ratio 0.20-0.39). This is the most striking finding:

- K562/Multi: 236 edges/feature at L0→L5
- TS/Multi: 75 edges/feature at L0→L5 (3.2x fewer)

**Interpretation:** The multi-tissue SAE features were annotated and selected based on K562 training data (activation_freq thresholds). These features may be less relevant or less frequently activated in non-K562 cell types, resulting in weaker causal effects.

### 2. Effect sizes are substantially weaker

- Mean |d| drops from 0.98 (K562/Multi) to 0.72 (TS/Multi)
- Only 10.4% of TS edges are strong (|d|>1) vs 34.4% for K562
- The model produces more diffuse, weaker perturbation effects on non-K562 cells

### 3. Even more inhibitory

TS/Multi circuits are **89.4% inhibitory** vs ~80% for both K562 conditions. When processing non-K562 cells, ablating features even more consistently reduces downstream activations. This suggests features encode even more "necessary" (non-redundant) information for non-training-domain cells.

### 4. Biological coherence is SAE-dependent, not cell-dependent

The shared ontology fraction is nearly identical between K562/Multi (68.8%) and TS/Multi (68.5%). This confirms that **biological coherence is a property of the SAE features**, not the input cells. Multi-tissue SAEs produce biologically coherent circuits regardless of what cells are processed.

### 5. Same top circuits, weaker effects

The same biological circuits appear in TS cells but with reduced effect sizes:

| Circuit | d (K562/Multi) | d (TS/Multi) |
|---------|---------------:|-------------:|
| L0 DNA Damage → L5 DNA Damage | -3.84 | -1.30 |
| L5 DNA Damage → L17 G2/M Transition | -2.66 | -0.81 |
| L5 Centromere → L17 G2/M Transition | -1.76 | -0.87 |

The circuits are preserved but attenuated — the model uses the same computational pathways but processes the information less strongly.

## Biological Interpretation

The 3-5x reduction in circuit density for TS cells likely reflects:

1. **Cell-type mismatch**: SAE features were selected by annotation quality from training on K562-enriched data. Features tuned to leukemia biology (DNA damage, cell cycle) may be less salient for normal tissue cells.

2. **Expression program differences**: K562 has very active cell cycle and DNA damage programs. Normal immune/kidney/lung cells have different dominant expression programs that may activate different (unselected) features more strongly.

3. **Activation sparsity**: Features optimized for K562 may fire less consistently across diverse cell types, reducing the statistical power of the causal analysis despite the same number of cells (200).

## Output Files

```
experiments/phase3_multitissue/circuit_tracing_ts_cells/
├── circuit_L00_features.json    — 30 source features from L0
├── circuit_L05_features.json    — 30 source features from L5
├── circuit_L11_features.json    — 30 source features from L11
├── circuit_graph.json           — 5,098 aggregated edges
└── circuit_analysis.json        — PMI comparison, biology, attenuation
```
