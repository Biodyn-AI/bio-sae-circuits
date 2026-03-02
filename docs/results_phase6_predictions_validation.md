# Phase 6: Gene-Level Predictions, Perturbation Validation & Disease Mapping — Results

**Script**: `src/15_phase6_predictions_and_validation.py`
**Output**: `experiments/phase6_predictions_validation/`
**Runtime**: 8.9 minutes
**Date**: 2026-03-02

## Overview

Phase 5 mapped 96,892 causal circuit edges at domain level. Phase 6 drills down to gene-level predictions, validates against CRISPRi perturbation data, and maps to disease biology.

---

## Step 1: Gene-Level Prediction Extraction

### Summary

| Metric | Value |
|--------|-------|
| Total edges processed | 96,892 |
| Edges with gene lists | 47,418 (48.9%) |
| Raw gene pairs extracted | 2,584,369 |
| Filtered predictions (≥2 edges or \|d\|>2) | **975,369** |

### Validation Against Known Biology

| Status | Count | Fraction |
|--------|-------|----------|
| **Confirmed** (STRING PPI or TRRUST) | 1,498 | 0.15% |
| — In STRING PPI (score≥700) | 1,329 | |
| — In TRRUST (TF→target) | 170 | |
| **Plausible** (≥2 shared GO terms) | 11,117 | 1.14% |
| **Novel** (no known link) | 962,754 | 98.7% |

### Context

| Property | Count |
|----------|-------|
| Consensus (cross-model) predictions | 319,534 (32.8%) |
| Novel domain pair predictions | 803,045 (82.3%) |

### Top Gene-Level Predictions

| Rank | Source → Target | Weighted \|d\| | Edges | Status |
|------|-----------------|---------------|-------|--------|
| 1 | VCP → TMEM208 | 12.74 | 1 | Novel |
| 2 | ATP5F1E → TMEM208 | 10.56 | 1 | Novel |
| 3 | CEP152 → CIP2A | 10.27 | 1 | Novel |
| 4 | MAPK1 → TOB2 | 7.02 | 1 | Novel |
| 5 | POLR2G → TGS1 | 6.89 | 1 | Novel |

**Key finding**: Of ~975K gene-pair predictions, only 0.15% match known protein-protein interactions or TF-target relationships, and 1.14% share GO annotations. The vast majority (98.7%) represent predictions connecting genes without established direct relationships. This is consistent with the models encoding broad co-expression patterns rather than direct regulatory links.

---

## Step 2: Perturbation Response Validation

### Setup

| Metric | Value |
|--------|-------|
| Replogle CRISPRi cells | 643,413 |
| Perturbation targets | 2,023 (+ 39,165 controls) |
| Overlap with circuit source genes | **599** |
| Gene pairs tested | **282,250** |

### Evaluation 1: Directional Accuracy

| Metric | Value |
|--------|-------|
| Concordant (same sign) | 159,138 (56.4%) |
| Discordant | 123,112 (43.6%) |

Marginally above chance (50%), consistent with prior evidence that models encode co-expression rather than causal regulation.

### Evaluation 2: Magnitude Correlation

| Metric | Value |
|--------|-------|
| Spearman ρ | 0.038 |
| p-value | 1.0 × 10⁻⁹² |

Statistically significant due to large N (282K pairs) but near-zero effect size. Circuit edge strength does not meaningfully predict perturbation response magnitude.

### Evaluation 3: Target Gene Enrichment

| Metric | Value |
|--------|-------|
| Source genes tested | 599 |
| Significantly enriched (p<0.05) | **36** (6.0%) |
| Median odds ratio | ~0.0 (most non-enriched) |

Top enriched source genes:

| Gene | OR | Responsive/Predicted |
|------|-----|---------------------|
| VMP1 | 9.13 | 20/959 |
| HSPA5 | 8.79 | 29/109 |
| CDT1 | 4.01 | 33/1,996 |
| PSMC2 | 2.37 | 136/1,785 |
| PSMC1 | 1.84 | 113/2,408 |

Only 6.0% of source genes show significant target enrichment — broadly consistent with Phase 2/3 findings (~6-10% TF specificity).

### Evaluation 4: Consensus vs All Predictions

| Subset | N pairs | Sign Accuracy | Spearman ρ |
|--------|---------|--------------|------------|
| All predictions | 282,250 | 56.4% | 0.038 |
| **Consensus only** | 104,443 | **57.3%** | **0.044** |
| Novel only | 242,670 | 56.5% | 0.039 |

Cross-model consensus predictions show marginally better sign accuracy (+0.9%) and magnitude correlation (+0.006), suggesting cross-model agreement captures slightly more genuine biology — but the improvement is minimal.

### Conclusion

As expected, the perturbation validation confirms that SAE circuit predictions are only weakly predictive of actual perturbation responses. The 56.4% sign accuracy (vs 50% chance) and 6% enrichment rate are consistent with the NMI paper finding that these models primarily encode co-expression, not causal regulatory relationships.

---

## Step 3: Disease Gene Mapping

### Disease Gene Set Definitions

Built from GO BP terms by keyword matching:

| Category | Genes | GO Terms |
|----------|-------|----------|
| DNA damage/repair | 165 | 24 |
| Cell cycle (cancer) | 177 | 29 |
| Apoptosis | 181 | 31 |
| Immune response | 183 | 36 |
| Metabolism (cancer) | 38 | 4 |
| Transcription regulation | 287 | 20 |
| Oncogenic signaling | 101 | 20 |
| Protein quality control | 166 | 23 |
| Angiogenesis | 29 | 6 |
| Metastasis/migration | 99 | 19 |
| TRRUST TFs | 795 | — |

### Domain-Disease Enrichment

| Metric | Value |
|--------|-------|
| Domains enriched for ≥1 disease category | **1,073/1,127** (95.2%) |
| Top-50 PageRank hubs enriched | **50/50** (100%) |

### Disease Domains Are More Central

| Group | Median Edge Count |
|-------|-------------------|
| Disease-enriched domains | 14 |
| Non-disease domains | 3 |
| Mann-Whitney p | **1.2 × 10⁻¹¹** |

Disease-associated domains are significantly more central in the circuit graph.

### Disease Circuit Paths

| Disease Category | Enriched Domains | Circuit Edges | Consensus Edges | Mean \|d\| |
|-----------------|------------------|---------------|-----------------|-----------|
| Transcription regulation | 739 | 28,155 | 7,046 | 1.11 |
| Immune response | 660 | 27,071 | 7,305 | 1.10 |
| Apoptosis | 631 | 25,622 | 6,614 | 1.11 |
| Cell cycle (cancer) | 577 | 25,180 | 6,610 | 1.12 |
| Protein quality control | 576 | 21,896 | 6,256 | 1.10 |
| DNA damage/repair | 532 | 22,233 | 6,010 | 1.12 |
| Oncogenic signaling | 513 | 15,921 | 3,826 | 1.09 |
| Metastasis/migration | 461 | 12,850 | 3,601 | 1.15 |
| Angiogenesis | 243 | 4,066 | 1,302 | 1.07 |
| Metabolism (cancer) | 231 | 8,775 | 3,724 | 1.15 |
| TRRUST TFs | 152 | 2,510 | 958 | 1.12 |

### Cross-Model Disease Validation

| Metric | Value |
|--------|-------|
| Disease pairs in consensus | 1,126/15,434 (7.3%) |
| Non-disease in consensus | 16/746 (2.1%) |
| **Fisher's OR** | **3.59** |
| **p-value** | **< 0.001** |

**Key finding**: Disease-relevant circuit pairs are 3.59× more likely to be cross-model consensus (p<0.001), suggesting that disease-relevant biology is especially well-captured by both models independently.

---

## Summary

| Analysis | Key Finding |
|----------|------------|
| Gene predictions | 975K gene pairs; 98.7% novel (no known link) |
| Perturbation validation | 56.4% sign accuracy (weak, as expected) |
| Disease mapping | 95% of domains disease-enriched; hubs more central (p<10⁻¹¹) |
| Disease × consensus | Disease pairs 3.59× enriched in cross-model consensus |

---

## Output Files

| File | Size | Contents |
|------|------|----------|
| `step1_gene_predictions.json` | 134 MB | 975K gene-pair predictions with validation |
| `step2_perturbation_validation.json` | 39 KB | Perturbation response validation results |
| `step3_disease_mapping.json` | 230 KB | Disease gene set enrichment + circuit mapping |
