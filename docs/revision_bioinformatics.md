# Bioinformatics revision (2026) — reviewer-concern ↔ experiment mapping

The first-round submission to *Bioinformatics* returned with 17 reviewer
points (5 major + 4 minor from Reviewer 1; 5 major + 2 minor from
Reviewer 2) plus an editor's request for an archival DOI. This document
cross-references each point to the experiment, text section, or figure
that addresses it.

## One-shot summary

| Reviewer point | Experiment(s) | Artifact location |
|---|---|---|
| **R1-M1** Non-immortalized / non-malignant validation cells | E1 (RPE1), E1b (Shifrut primary CD8⁺ T) | `experiments/revision/E1_nonimmortalized_validation/`, `E1b_shifrut_primary_tcell/` |
| **R1-M1** Guide-efficacy filter | E2 (pool-level FDR + log₂FC) | Applied inside E1 and E1b |
| **R1-M2** Direct vs indirect targets via ChIP-seq | E3 | `E3_direct_indirect/run.py` |
| **R1-M3** STRING/TRRUST vs matched-cell-type ChIP-seq | E4 (full + K562-restricted) | `E4_chipseq_coherence/run_chipseq.py` + `k562_restricted.py` |
| **R1-M4** §3.4 factual corrections (MLM, relative rank) | Text only | Paper §3.4 rewritten |
| **R1-M5** Input-size normalisation | E5 | `E5_input_size_normalization/run.py` |
| **R1-m1/m2** Figure 10 (cross-model) caption + legend | Fig regen | `paper/generate_phase4_figures.py` |
| **R1-m3** Soften "confirms" language | Reframe | Abstract + §Results rewritten |
| **R1-m4** Abstract term clarity + chance baselines | E12 + inline defs | Abstract rewritten; `E12_permutation_baselines/` |
| **R2-M1** Novelty + annotation bias + reorganised co-expression? | Novelty paragraph, E6, E7 | §1 "Contribution"; `E6_random_features/`; `E7_partial_correlation/` |
| **R2-M2** Causal language vs weak CRISPRi | Sign-bias correction; E3 | `src/*` + `E1/E3` |
| **R2-M3** 200-cell design fragile | E9, E10 | `E9_bootstrap_stability/`, `E10_per_celltype/` |
| **R2-M4** Model-level vs biological causality | "Two notions of causality" paragraph | Paper §1 |
| **R2-M5** \|d\|>0.5 arbitrary; sensitivity | E11 | `E11_threshold_sweep/` |
| **R2-m1** Pipeline schematic | F1 | `F1_schematic/draw_schematic.py` |
| **R2-m2** "wiring diagram" / "complete pathways" overstated | Language audit | Paper, all occurrences removed |
| **Editor** Archival DOI | Zenodo | Done on acceptance |

## Re-tracing commands (for E6, E9, E10)

The three experiments that require re-running circuit tracing with modified
source-feature selection or cell subsets use flags added to the main tracing
script:

```bash
# E6: random source-feature seeds (two independent draws)
python src/13_causal_circuit_tracing.py \
    --source-layers 0 --n-features 20 --n-cells 50 \
    --random-feature-seed 1 \
    --out-dir experiments/revision/E6_random_features/seed1

python src/13_causal_circuit_tracing.py \
    --source-layers 0 --n-features 20 --n-cells 50 \
    --random-feature-seed 2 \
    --out-dir experiments/revision/E6_random_features/seed2
```

```bash
# E9: bootstrap-style sample-size sweep
for N in 50 100; do
    python src/13_causal_circuit_tracing.py \
        --source-layers 0 --n-features 20 --n-cells $N \
        --out-dir experiments/revision/E9_bootstrap_stability/N$N
done
```

```bash
# E10: per-immune-cell-type (multi-tissue SAEs at layers {0, 5, 11, 17})
for TYPE in "B cell" "CD4-positive, alpha-beta T cell" "macrophage"; do
    OUT="experiments/revision/E10_per_celltype/$(echo $TYPE | tr ' ,' '__')"
    python src/13_causal_circuit_tracing.py \
        --source-layers 0 --n-features 20 --n-cells 50 \
        --data-source tabula_sapiens \
        --ts-tissue immune --ts-cell-type "$TYPE" \
        --sae-dir experiments/phase3_multitissue/sae_models \
        --available-layers 0,5,11,17 \
        --out-dir "$OUT"
done
```

After each set of retraces completes, run the analysis script inside the
corresponding sub-folder (`analyze.py` for E6 / E9 / E10).

## Dataset accessions

| Dataset | Accession / source | Notes |
|---|---|---|
| Replogle 2022 K562 + RPE1 Perturb-seq | GEO **GSE264667** | Used for circuit tracing input cells (K562 control subset) and gene-level CRISPRi validation on both K562 and RPE1 arms. |
| Shifrut 2018 primary human CD8⁺ T-cell CRISPR-KO | GEO **GSE119450** | E1b validation; CRISPR-KO (not CRISPRi), so efficacy filter uses a KO-scale threshold. |
| Tabula Sapiens v1.0 | figshare / CELL×GENE | Immune (`immune_subset_20000`), kidney, lung arms used for TS circuit conditions and E10. |
| ENCODE Uniform TFBS clusters V3 | UCSC track `wgEncodeRegTfbsClusteredV3` + `InputsV3.tab` | E3 direct-target partition and E4 ChIP-seq enrichment. |

## Random seeds used

| Seed | Purpose |
|---|---|
| `0` | NumPy default generator for Replogle / Shifrut control-cell subsampling (E1, E1b, E3, E7). |
| `42` | Stratified Tabula Sapiens cell sampling across immune / kidney / lung. |
| `1`, `2` | E6 random source-feature draws (two independent samples). |

All pipeline code is deterministic given these seeds modulo MPS
non-determinism in a small number of matmul reductions.
