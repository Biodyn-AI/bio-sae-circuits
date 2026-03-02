# Subproject 42: Phase 4 Multi-Tissue — Causal Feature-to-Feature Circuit Tracing

*Completed: 2026-03-01*

## Overview

Extends Phase 4 causal circuit tracing to **multi-tissue SAEs** trained in Phase 3 (K562 + Tabula Sapiens immune/kidney/lung cells). Multi-tissue SAEs were trained at layers 0, 5, 11, 17 only — so downstream tracing is restricted to these 4 layers.

## Configuration

**Script:** `13_causal_circuit_tracing.py` (same script with `--sae-dir`, `--available-layers` flags)
**SAE models:** `experiments/phase3_multitissue/sae_models/layer{00,05,11,17}_x4_k32/`
**Source layers:** L0, L5, L11 (L17 has no downstream SAE layers)
**Downstream tracing:** Restricted to layers with SAEs (L5, L11, L17)
**Cells:** 200 control cells from Replogle K562 CRISPRi dataset (same cells as K562-only)
**Compute:** 3.5 hours on Apple Silicon (MPS), 18,465 forward passes total

## Per-Source-Layer Results

| Source | Downstream | Forward Passes | Time | Total Edges | Avg/Feature |
|--------|-----------|---------------:|-----:|------------:|------------:|
| L0 | L5,L11,L17 | 6,066 | 74 min | 13,630 | 454 |
| L5 | L11,L17 | 6,199 | 71 min | 6,465 | 216 |
| L11 | L17 | 6,200 | 66 min | 4,009 | 134 |

## Aggregate Circuit Graph

- **8,298 significant causal edges** (|d| > 0.5, consistency > 0.7)
- **90 source features** → **4,171 unique target features**
- Mean |Cohen's d| = 0.98, median = 0.87, max = 13.18
- 34.4% of edges have |d| > 1.0 (strong effects)
- 2.7% have |d| > 2.0 (very strong effects)
- **79.9% inhibitory** — nearly identical to K562-only (80.1%)

## Comparison with K562-Only Circuit Tracing

### Per-Layer-Pair Comparison (Apples to Apples)

Comparing the same layer pairs between K562-only and multi-tissue SAEs:

| Layer Pair | K562 Edges/Feature | Multi Edges/Feature | Ratio |
|------------|-------------------:|--------------------:|------:|
| L0→L5 | 200.7 | 236.3 | 1.18 |
| L0→L11 | 122.7 | 147.2 | 1.20 |
| L0→L17 | 58.5 | 70.9 | 1.21 |
| L5→L11 | 119.1 | 140.5 | 1.18 |
| L5→L17 | 61.9 | 75.0 | 1.21 |
| L11→L17 | 145.2 | 133.6 | 0.92 |

**Key finding:** Multi-tissue SAE features show **18-21% MORE significant causal edges** per feature at most layer pairs (ratio 1.18-1.21). The exception is L11→L17 where K562-only has slightly more edges (ratio 0.92). This suggests multi-tissue features are more broadly connected — possibly because they capture more universal biological programs.

### Effect Size Distribution

| Threshold | K562-only | Multi-tissue |
|-----------|----------:|-------------:|
| |d| > 0.5 | 100% | 100% |
| |d| > 1.0 | 41.4% | 34.4% |
| |d| > 2.0 | 4.3% | 2.7% |
| |d| > 3.0 | 1.0% | 0.5% |
| |d| > 5.0 | 0.2% | 0.1% |

Multi-tissue edges are slightly weaker on average (more edges but fewer extreme effects). This is consistent with multi-tissue features being more general/diffuse rather than cell-type-specific.

### Biological Coherence — Multi-Tissue is STRONGER

| Metric | K562-only | Multi-tissue |
|--------|----------:|-------------:|
| Annotated edge pairs | 31,176 | 4,591 |
| Shared ontology terms | 16,507 (52.9%) | 3,157 (68.8%) |

**68.8% of multi-tissue causal edges share ontology terms** between source and target, compared to 52.9% for K562-only. This is a substantial improvement (+16 percentage points), suggesting that multi-tissue SAE features capture cleaner, more biologically coherent circuits.

### Inhibitory/Excitatory Balance

| | K562-only | Multi-tissue |
|-|----------:|-------------:|
| Inhibitory | 80.1% | 79.9% |
| Excitatory | 19.9% | 20.1% |

Virtually identical — the fundamental computational architecture (predominantly inhibitory information flow) is consistent across SAE training regimes.

## Top Hub Features (Multi-tissue)

**Highest out-degree (most downstream targets):**

| Feature | Biology | Out-Degree |
|---------|---------|----------:|
| L0_F2569 | Histone Modification | 2,146 |
| L0_F996 | RNA Processing | 1,273 |
| L0_F2558 | Dephosphorylation | 609 |
| L0_F2551 | DNA Damage Response | 562 |
| L0_F2656 | Aerobic Electron Transport | 514 |

**Highest in-degree (most upstream sources):**

| Feature | In-Degree |
|---------|----------:|
| L17_F3225 | 58 |
| L17_F2938 | 55 |
| L11_F886 | 39 |
| L17_F2252 | 36 |
| L17_F3071 | 36 |

## Top Interpretable Circuits

| Source | Target | d | Shared Biology |
|--------|--------|--:|------|
| L0_F2551 DNA Damage Response → | L5_F3538 DNA Damage Response | -3.84 | Mitotic regulation, spindle checkpoint, G2/M |
| L5_F3538 DNA Damage Response → | L17_F1269 G2/M Transition | -2.66 | Sister chromatid separation, p53 signaling |
| L5_F3098 Centromere Assembly → | L17_F1269 G2/M Transition | -1.76 | APC/C activators, spindle checkpoint |
| L5_F3538 DNA Damage Response → | L11_F3296 G2/M Transition | -1.57 | Prometaphase, chromatid segregation |
| L5_F3098 Centromere Assembly → | L11_F3296 G2/M Transition | -1.37 | Nuclear division, kinetochore |

## Attenuation Analysis

Average significant downstream edges per layer distance:

| Source | +5 layers | +11 layers | +17 layers |
|--------|----------:|-----------:|-----------:|
| L0 | 236 | 147 | 71 |
| L5 | 140 | 75 | — |
| L11 | 134 | — | — |

Clear decay with distance: L0 effects drop ~70% from +5 to +17 layers.

## PMI Comparison

PMI data was computed for K562-only SAEs (Phase 2, Step 11). Using K562 PMI edges as reference:

| Layer Pair | PMI Edges | Causal Edges | Overlap |
|------------|----------:|-------------:|--------:|
| L0→L5 | 25,000 | 1,455 | 0 (0%) |
| L5→L11 | 25,000 | 1,466 | 2 (0.1%) |
| L11→L17 | 25,000 | 1,500 | 5 (0.3%) |

Near-zero overlap is expected — PMI was computed on K562-only SAEs (different feature space than multi-tissue SAEs). Different SAE training produces different feature dictionaries, so edge identity cannot be compared directly.

## Key Findings

1. **Multi-tissue circuits are more biologically coherent**: 68.8% shared ontology vs 52.9% for K562-only. Training on diverse cell types produces features that form tighter biological circuits.

2. **Multi-tissue features are more broadly connected**: 18-21% more significant downstream edges per feature at most layer pairs. Multi-tissue features may capture more universal pathways.

3. **Same computational architecture**: ~80% inhibitory ratio and similar effect size distributions — the model's fundamental information flow patterns are SAE-independent.

4. **Slightly weaker individual effects**: Multi-tissue has fewer extreme effect sizes (34.4% |d|>1 vs 41.4%), consistent with features being more general/diffuse.

5. **DNA damage/cell cycle circuits dominate**: Multi-tissue features are heavily enriched for DNA damage response and cell cycle pathways, reflecting the K562 cell context (leukemia cell line) even when SAEs were trained on diverse tissues.

## Output Files

```
experiments/phase3_multitissue/circuit_tracing/
├── circuit_L00_features.json    — 30 source features, 13,630 edges
├── circuit_L05_features.json    — 30 source features, 6,465 edges
├── circuit_L11_features.json    — 30 source features, 4,009 edges
├── circuit_graph.json           — aggregated graph + hub analysis
└── circuit_analysis.json        — PMI comparison, biology, attenuation
```
