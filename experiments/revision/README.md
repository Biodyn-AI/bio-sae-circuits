# Revision experiments (Bioinformatics major revision, 2026)

This directory contains the twelve stand-alone analysis scripts (plus one
schematic-figure generator) written to address the reviewer concerns on the
initial *Bioinformatics* submission. Each sub-folder is self-contained and
runs from existing circuit outputs plus public datasets.

## Environment

All scripts use the same Python environment as the main pipeline (see the
top-level `requirements.txt`). Paths are configurable via the environment
variables listed below:

| Variable | Default | Purpose |
|---|---|---|
| `BIO_SAE_ROOT` | `../..` | Anchor to this repository's root (relative to each script). |
| `REPLOGLE_H5AD` | `../../data/replogle_concat.h5ad` | Replogle 2022 K562 + RPE1 CRISPRi `AnnData` file. |
| `SHIFRUT_H5AD` | `../../data/shifrut_primary_cd8_tcell_cropseq.h5ad` | Shifrut 2018 primary human CD8⁺ T-cell CRISPR-KO. |
| `ENCODE_DIR` | `../../data/encode` | Directory with ENCODE `encode_tf_targets_5celllines_edges.tsv` and `wgEncodeRegTfbsClusteredInputsV3.tab`. |
| `TS_DIR` | `../../data/tabula_sapiens` | Tabula Sapiens tissue `.h5ad` files. |

Dataset accessions are listed in `docs/revision_bioinformatics.md`.

## Script index

Each experiment ID maps to a specific reviewer concern. The mapping is in
`docs/revision_bioinformatics.md`; this README lists what each script does.

| ID | Folder / entry script | Purpose |
|---|---|---|
| **E1** | `E1_nonimmortalized_validation/run_validation.py` | CRISPRi validation of circuit predictions on K562 + non-cancer RPE1 arms of the Replogle screen, with pool-level guide-efficacy filter and sign-bias correction. |
| **E1b** | `E1b_shifrut_primary_tcell/run.py` | Follow-up validation on Shifrut 2018 primary human CD8⁺ T-cell CRISPR-KO — the truly non-immortalized, non-malignant control. Transparently null (documented). |
| **E3** | `E3_direct_indirect/run.py` | Partition validation gene pairs into ChIP-seq-supported direct vs. indirect targets using ENCODE K562 TF→target edges, and re-score directional accuracy on each partition. |
| **E4** | `E4_chipseq_coherence/run_chipseq.py` + `k562_restricted.py` | Fisher's-exact enrichment of circuit TF→target predictions against (i) 5-cell-line ENCODE ChIP-seq edges, and (ii) the K562-restricted subset (100 TFs with K562 experiments in `InputsV3`). |
| **E5** | `E5_input_size_normalization/run.py` | Input-size-normalised cross-model comparison: recomputes mean \|d\| per condition under feature-share and input-share normalisations, plus paired gene-pair analysis on 33 k common pairs. |
| **E6** | `E6_random_features/analyze.py` | Cross-checks annotation-selected vs randomly-sampled source features (two seeds) to quantify the annotation-selection contribution to coherence numbers. Relies on `src/13_causal_circuit_tracing.py --random-feature-seed` for the retracing. |
| **E7** | `E7_partial_correlation/run_partial_corr.py` | Tests whether circuit \|d\| is reducible to marginal driver-gene co-expression on the same cells (Pearson + Spearman regression). |
| **E9** | `E9_bootstrap_stability/analyze.py` | Cross-N sample-size stability (N ∈ {50, 100, 200}) via pairwise Jaccard + Pearson r on the L0 sub-graph. Relies on `src/13_causal_circuit_tracing.py` for re-tracing at smaller N. |
| **E10** | `E10_per_celltype/analyze.py` | Per-immune-cell-type (B cell, CD4⁺ T, macrophage) circuit stability using multi-tissue SAEs. Uses `src/13_causal_circuit_tracing.py --ts-tissue --ts-cell-type`. |
| **E11** | `E11_threshold_sweep/sweep.py` + `fdr_threshold.py` | Magnitude sweep of the `\|d\|` threshold and BH-FDR-controlled equivalent. |
| **E12** | `E12_permutation_baselines/compute_nulls.py` | Configuration-preserving permutation nulls for shared-ontology and inhibitory metrics, per condition. |
| **F1** | `F1_schematic/draw_schematic.py` | Re-draws the pipeline schematic used as main-text Figure 1. |

## Execution order

The analytical experiments (E7, E11, E12, E4, E5) run directly from existing
circuit outputs. The retracing experiments (E6 on random seeds; E9 at N ∈
{50, 100}; E10 per cell type) require running `src/13_causal_circuit_tracing.py`
with the appropriate flags before the analysis scripts can aggregate results.
Exact command lines are in `docs/revision_bioinformatics.md`.

## Outputs

Every script writes its aggregate JSON output alongside itself (e.g.
`E7_partial_correlation/results.json`, `E4_chipseq_coherence/chipseq_coherence.json`,
etc.). These JSONs are consumed by `paper/generate_phase4_figures.py` and by
the manuscript's result paragraphs directly.
