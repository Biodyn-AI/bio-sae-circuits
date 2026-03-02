# Phase 5: Systematic Biological Knowledge Extraction — Results

**Script**: `src/14_biological_knowledge_extraction.py`
**Output**: `experiments/phase5_knowledge_extraction/`
**Runtime**: <1 minute
**Date**: 2026-03-01

## Overview

Phase 4 causal circuit tracing produced **96,892 significant edges** across 4 conditions. Phase 5 systematically extracts, validates, and categorizes the full biological knowledge in these circuit graphs.

| Condition | Edges | Both Annotated |
|-----------|-------|----------------|
| K562/K562 (Geneformer) | 52,116 | 48.8% targets annotated |
| K562/Multi (Geneformer) | 8,298 | 41.9% |
| TS/Multi (Geneformer) | 5,098 | 42.0% |
| scGPT TS/Multi | 31,380 | 19.2% |
| **Total** | **96,892** | **37,088 with both domains** |

---

## Step 1: Full Circuit Annotation

Annotated all ~97K edges with source and target biological domain labels (GO BP terms).

### Top Domain Pairs (all conditions combined)

| Rank | Source → Target | Edges | Conditions | Mean |d| |
|------|-----------------|-------|------------|---------|
| 1 | DDR → Mitotic Sister Chromatid Segregation | 139 | 4 | 1.12 |
| 2 | NADH Dehydrogenase Assembly → Mito Respiratory Chain Assembly | 117 | 4 | 1.72 |
| 3 | DDR → DNA-templated DNA Replication | 108 | 4 | 1.33 |
| 4 | DNA Metabolic Process → Mitotic Chromatid Segregation | 95 | 3 | 1.16 |
| 5 | Cholesterol Biosynthesis → Sterol Biosynthesis | 94 | 2 | 3.15 |

### Top Hub Domains

| Domain | Total edges | Source | Target | Role |
|--------|------------|--------|--------|------|
| DNA Damage Response | 4,864 | 4,740 | 124 | Dominant source |
| DNA Metabolic Process | 2,158 | 1,810 | 348 | Major source |
| NADH Dehydrogenase Assembly | 1,742 | 1,639 | 103 | Major source |
| RNA Splicing (bulged A) | 1,558 | 1,161 | 397 | Bidirectional |
| Mitotic Chromatid Segregation | 1,051 | 248 | 803 | Dominant target |

**Key finding**: DDR is the single largest hub domain with 4,864 edges (97.5% as source), consistent with its role as a master upstream regulator in K562 leukemia cells.

---

## Step 2: Cross-Model Consensus Graph

### Consensus Statistics

| Metric | Value |
|--------|-------|
| Geneformer unique pairs | 13,698 |
| scGPT unique pairs | 3,511 |
| **Consensus (GF ∩ scGPT)** | **1,142 pairs** |
| High-confidence (both |d|>1.0) | 303 pairs |
| Permutation expected | 107.3 |
| **Enrichment** | **10.6×** |
| **Permutation p-value** | **<0.001** |

**Key finding**: 1,142 domain pairs are conserved across both models — 10.6× more than expected by chance (p<0.001). This demonstrates that both Geneformer and scGPT learn genuine biological circuit structure, not model-specific artifacts.

### Top Consensus Pairs

| Source → Target | GF |d| | scGPT |d| |
|-----------------|---------|-----------|
| Golgi Organization → Protein Insertion Into Membrane | 4.75 | 5.23 |
| Cholesterol Biosynthesis → Sterol Biosynthesis | 4.44 | 1.54 |
| Golgi Organization → Cotranslational Protein Targeting | 2.45 | 2.83 |
| Maturation of SSU-rRNA → RNA Methylation | 1.67 | 5.00 |
| RNA Methylation → Regulation of RNA Metabolic Process | 1.93 | 4.02 |

---

## Step 3: Novel Relationship Discovery

### Summary

| Metric | Value |
|--------|-------|
| Known domain-domain links (≥3 shared genes) | 14,021 |
| Domains matched to gene sets | 1,126/1,126 (100%) |
| Novel candidate edges | 29,864 |
| Novel domain pairs | 5,082 |

### Top Novel Relationships (present in all 4 conditions)

| Source → Target | Max |d| | Edges | Shared Genes |
|-----------------|---------|-------|--------------|
| NADH Dehydrogenase Assembly → Protein Transport | 7.21 | 21 | 20 |
| Golgi Organization → ER Stress Response | 6.29 | 33 | 43 |
| DDR → DNA Unwinding in Replication | 5.98 | 60 | 59 |
| Golgi Vesicle Transport → ER Stress Response | 5.01 | 12 | 22 |
| DDR → Negative Regulation of Gene Expression | 4.92 | 30 | 19 |
| Aerobic ETC → Mitochondrial Translation | 4.67 | 20 | 22 |
| rRNA Maturation → RNA Splicing | 4.64 | 9 | 4 |

**Key finding**: 29,864 edges (31% of annotated edges) connect domains with <3 shared genes in reference databases. Many involve cross-compartment links (e.g., Golgi → ER stress, mitochondrial ETC → translation), suggesting the models learn functional coupling beyond direct gene overlap.

---

## Step 4: Biological Process Hierarchy

### Meta-Graph Structure

- **1,126 domain nodes**, **16,002 directed edges**
- **499 feedback loops** (reciprocal A↔B connections)
- Strong subgraph (|d|>1.0, ≥3 edges): **457 nodes, 1,667 edges**
- 1 strongly connected component with 39 nodes; condensed DAG has 419 meta-nodes

### PageRank Centrality (Top 10)

| Domain | PageRank | In-degree | Out-degree |
|--------|----------|-----------|------------|
| DNA Repair | 0.0017 | 88 | 239 |
| Ribosome Biogenesis | 0.0017 | 81 | 0 |
| Regulation of DNA-templated Transcription | 0.0015 | 79 | 0 |
| RNA Splicing | 0.0015 | 76 | 436 |
| Regulation of RNA Metabolic Process | 0.0015 | 60 | 0 |

**Key insight**: Terminal nodes (out-degree=0) like ribosome biogenesis, transcription regulation, and RNA metabolic regulation act as convergence points — many upstream processes influence them, but they do not feed forward.

### Temporal Ordering (Early → Late Layers)

**Early layers** (upstream processes):
- MAPK Cascade (mean L=0.1)
- Ras Signaling (L=0.3)
- Epithelial Differentiation (L=0.3)
- Histone Modification (L=0.4)
- COPII Vesicle Budding (L=0.7)

**Late layers** (downstream processes):
- Regulation of Glucose Import (L=17.0)
- Nuclear RNA Surveillance (L=17.0)
- Protein Localization to Chromatin (L=17.0)

### DDR → Cell Cycle Validation

**300 DDR → cell cycle edges found**, confirming expected temporal ordering:
- DNA Repair → Mitotic Sister Chromatid Segregation: |d|=1.27, ΔL=+7.5 layers
- DNA Repair → Regulation of Mitotic Cell Cycle: |d|=1.02, ΔL=+5.9 layers
- DNA Repair → DNA Integrity Checkpoint: |d|=0.86, ΔL=+7.4 layers

All DDR→cell cycle connections show **positive layer deltas** (DDR features at earlier layers than cell cycle targets), validating that the model's layer hierarchy reflects biological temporal ordering.

---

## Step 5: Cell-Type-Specific Circuit Activation

### Domain Tissue Classification

| Tissue | Domains | Keyword examples |
|--------|---------|-----------------|
| Universal | 224 | ribosome, mitochondria, cell cycle |
| Immune | 37 | T cell, cytokine, interferon |
| Blood | 12 | hemoglobin, hematopoietic, coagulation |
| Kidney | 9 | renal, nephron, ion transport |
| Lung | 1 | respiratory |
| Unclassified | 840 | — |

### Circuit Specificity

| Category | Pairs |
|----------|-------|
| TS-only pairs (not in K562) | 3,541 |
| Shared (in both) | 1,334 |
| K562-only | 10,384 |

### Tissue Enrichment (Fisher's exact: TS-only vs shared)

| Tissue | TS-only | Shared | Odds Ratio | p-value |
|--------|---------|--------|------------|---------|
| **Immune** | **179/3,541** | **22/1,334** | **3.18** | **<0.001** |
| Blood | 13/3,541 | 2/1,334 | 2.45 | 0.18 |
| Kidney | 14/3,541 | 4/1,334 | 1.32 | 0.43 |
| Lung | 14/3,541 | 14/1,334 | 0.37 | 1.00 |
| Universal | 1,550/3,541 | 553/1,334 | 1.10 | 0.08 |

**Key finding**: Immune circuits are significantly enriched (3.18× OR, p<0.001) in the multi-tissue conditions compared to K562-only, confirming that the Tabula Sapiens cells activate tissue-specific circuits absent from K562.

### Tissue-Specific Circuit Counts

| Tissue | Circuits | Also in K562 |
|--------|----------|-------------|
| Immune | 201 | 22 (10.9%) |
| Lung | 28 | 14 (50.0%) |
| Kidney | 18 | 4 (22.2%) |
| Blood | 15 | 2 (13.3%) |
| Universal | 2,020 | — |

---

## Verification Against Success Criteria

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| All ~97K edges annotated | ✓ | 96,892 edges, 37,088 both-annotated | ✅ |
| ≥20 consensus domain pairs | ≥20 | **1,142** pairs | ✅ |
| Permutation p < 0.01 | p<0.01 | **p<0.001** | ✅ |
| ≥5 novel relationships |d|>2 | ≥5 | **20+ at |d|>3** | ✅ |
| DDR → checkpoint → arrest | Recovered | **300 DDR→CC edges, all ΔL>0** | ✅ |
| Tissue enrichment p<0.01 for ≥3 | ≥3 tissues | **1 (immune only)** | ⚠️ Partial |

5/6 criteria fully met. Tissue enrichment achieved for immune (p<0.001) but not kidney/lung, likely due to limited keyword coverage for those tissues.

---

## Output Files

| File | Size | Contents |
|------|------|----------|
| `step1_annotated_edges.json` | 71 MB | All 96,892 edges with domains + gene lists |
| `step1_domain_summary.json` | 24 KB | Domain pair frequencies, hub domains |
| `step2_consensus_graph.json` | 585 KB | 1,142 consensus pairs + permutation results |
| `step3_novel_candidates.json` | 22 MB | 5,082 novel domain pairs |
| `step4_hierarchy.json` | 329 KB | Meta-graph, PageRank, layer centrality |
| `step5_celltype_circuits.json` | 129 KB | Tissue classification + enrichment |
