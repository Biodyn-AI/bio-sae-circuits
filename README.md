# Causal Circuit Tracing in Single-Cell Foundation Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the causal circuit tracing pipeline for discovering how sparse autoencoder (SAE) features causally drive each other across network depth in single-cell foundation models. We apply feature-to-feature ablation to **Geneformer V2-316M** and **scGPT whole-human**, discovering 96,892 significant causal edges, extracting systematic biological knowledge, and validating predictions against genome-scale CRISPRi perturbation data.

This is a **companion repository** to [bio-sae](https://github.com/Biodyn-AI/bio-sae), which contains the SAE training pipeline and interactive feature atlases (Phases 1--3). This repository covers **Phases 4--6**: circuit tracing, knowledge extraction, and gene-level validation.

## Key Findings

- **96,892 significant causal edges** across four experimental conditions (3 Geneformer + 1 scGPT), totaling 61,736 forward passes
- **Inhibitory dominance**: 65--89% of causal edges are inhibitory, indicating features encode necessary rather than redundant information
- **~53% biological coherence**: over half of causal edges connect features sharing ontology annotations---invariant across model architecture, SAE training data, and input cell type
- **Architectural divergence**: scGPT produces stronger individual effects (mean |d|=1.40 vs 1.05) with more balanced excitatory/inhibitory dynamics (65/35 vs 80/20)
- **1,142 cross-model consensus domain pairs**: 10.6x enrichment over chance (p < 0.001), demonstrating both models independently converge on the same biological circuit structure
- **29,864 novel edges** connecting biologically unlinked domains, with top candidates in cross-compartment functional coupling (mitochondria-to-cytoplasm, Golgi-to-ER stress)
- **Validated process hierarchy**: DNA damage response at early layers, gene expression regulation at late layers; 300 DDR->cell cycle edges all show correct temporal ordering
- **Disease-relevant biology is preferentially conserved**: disease-associated domains are 3.59x more likely to be cross-model consensus (p < 0.001) and significantly more central in the circuit graph (p = 1.2 x 10^-11)
- **Weak gene-level causal predictions**: 56.4% directional accuracy against CRISPRi (vs 50% chance), confirming models encode co-expression, not causal regulation---consistent with the companion study

## Repository Structure

```
bio-sae-circuits/
├── README.md                     # This file
├── LICENSE                       # MIT license
├── requirements.txt              # Python dependencies
│
├── src/                          # Geneformer analysis pipeline
│   ├── sae_model.py              # Core TopK SAE architecture (shared with bio-sae)
│   ├── 13_causal_circuit_tracing.py       # Phase 4: Feature-to-feature circuit tracing
│   ├── 14_biological_knowledge_extraction.py  # Phase 5: Systematic knowledge extraction
│   └── 15_phase6_predictions_and_validation.py # Phase 6: Gene predictions + validation
│
├── scgpt_src/                    # scGPT analysis pipeline
│   └── 13_causal_circuit_tracing.py       # Phase 4: scGPT circuit tracing
│
├── paper/                        # Figure generation scripts
│   ├── generate_phase4_figures.py # Circuit tracing figure generation
│   └── generate_phase5_figures.py # Knowledge extraction figure generation
│
├── experiments/revision/         # Bioinformatics revision (2026) experiments E1–E12 + F1
│   └── ...                       # (self-contained; see its own README.md)
│
└── docs/                         # Detailed results documentation
    ├── results_phase4_circuit_tracing.md   # Geneformer K562/K562
    ├── results_phase4_multitissue.md       # Geneformer K562/Multi-tissue
    ├── results_phase4_ts_cells.md          # Geneformer TS/Multi-tissue
    ├── results_phase4_scgpt.md             # scGPT TS/Multi-tissue
    ├── results_phase5_knowledge_extraction.md  # Systematic extraction
    ├── results_phase6_predictions_validation.md # Gene predictions + disease mapping
    └── revision_bioinformatics.md          # Reviewer-concern ↔ experiment mapping (2026 revision)
```

## Pipeline Overview

### Phase 4: Causal Circuit Tracing

**Script**: `src/13_causal_circuit_tracing.py` (Geneformer), `scgpt_src/13_causal_circuit_tracing.py` (scGPT)

For each selected source feature, the algorithm:
1. Runs a clean forward pass, capturing hidden states at all layers
2. Encodes the source layer through the SAE, zeros the target feature, decodes back
3. Propagates the ablated hidden state through all downstream layers
4. Encodes each downstream layer through its SAE, measuring feature activation changes
5. Computes Cohen's d and consistency across 200 cells per condition

**Four experimental conditions:**

| Condition | Model | SAE | Cells | Edges |
|-----------|-------|-----|-------|-------|
| K562/K562 | Geneformer | K562-trained | K562 | 52,116 |
| K562/Multi | Geneformer | Multi-tissue | K562 | 8,298 |
| TS/Multi | Geneformer | Multi-tissue | Tabula Sapiens | 5,098 |
| scGPT TS/Multi | scGPT | TS-trained | Tabula Sapiens | 31,380 |

### Phase 5: Systematic Biological Knowledge Extraction

**Script**: `src/14_biological_knowledge_extraction.py`

Annotates all 96,892 edges with GO Biological Process domain labels, then performs:
1. **Cross-model consensus**: identifies 1,142 domain pairs conserved across models
2. **Novel relationship discovery**: finds 29,864 edges connecting unlinked domains
3. **Process hierarchy**: reconstructs temporal ordering from layer positions
4. **Tissue-specific circuits**: identifies immune-specific circuits enriched 3.18x

### Phase 6: Gene-Level Predictions and Validation

**Script**: `src/15_phase6_predictions_and_validation.py`

1. **Gene-pair extraction**: 975,369 gene-level predictions from circuit edges
2. **CRISPRi validation**: tests against Replogle genome-scale perturbation screen (643K cells, 2,023 knockdowns)
3. **Disease gene mapping**: maps disease gene sets onto circuit hubs; finds disease domains are more central and more conserved across models

## Prerequisites

### From bio-sae (companion repository)

This repository requires SAE checkpoints and feature annotations generated by [bio-sae](https://github.com/Biodyn-AI/bio-sae):

- **Trained SAE models**: `experiments/{phase}/sae_models/layer{NN}_x4_k32/sae_final.pt`
- **Feature annotations**: `experiments/{phase}/sae_models/layer{NN}_x4_k32/feature_annotations.json`
- **Feature catalogs**: `experiments/{phase}/sae_models/layer{NN}_x4_k32/feature_catalog.json`

### External Data

- **Geneformer V2-316M**: [HuggingFace ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)
- **scGPT whole-human**: [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)
- **Replogle CRISPRi**: [Replogle et al. (2022)](https://doi.org/10.1016/j.cell.2022.05.013) K562 genome-scale Perturb-seq
- **Tabula Sapiens**: [The Tabula Sapiens Consortium (2022)](https://doi.org/10.1126/science.abl4896)
- **Biological databases**: GO, KEGG, Reactome gene sets; STRING PPI; TRRUST TF-target edges

### Software

```bash
conda create -n bio-sae python=3.10
conda activate bio-sae
pip install -r requirements.txt
```

## Usage

All scripts contain a `BASE` path variable at the top that must be configured for your directory structure. See the configuration section at the top of each script.

```bash
# Phase 4: Causal circuit tracing (Geneformer)
# Requires: trained SAE models, extracted activations, Geneformer model
python src/13_causal_circuit_tracing.py \
    --sae-dir experiments/phase1_k562/sae_models \
    --available-layers 0,1,2,...,17 \
    --data-source k562

# Phase 4: Causal circuit tracing (scGPT)
python scgpt_src/13_causal_circuit_tracing.py

# Phase 5: Biological knowledge extraction
# Requires: circuit_graph.json from all 4 Phase 4 runs
python src/14_biological_knowledge_extraction.py

# Phase 6: Gene-level predictions and validation
# Requires: Phase 5 outputs + Replogle CRISPRi data
python src/15_phase6_predictions_and_validation.py
```

## Output Data

Phase 4--6 produce the following outputs (not included in this repository due to size):

| Output | Size | Description |
|--------|------|-------------|
| `circuit_graph.json` (x4) | ~100 MB total | Significant causal edges per condition |
| `step1_annotated_edges.json` | 71 MB | All 96,892 edges with domain labels + gene lists |
| `step2_consensus_graph.json` | 585 KB | 1,142 cross-model consensus pairs |
| `step3_novel_candidates.json` | 22 MB | 5,082 novel domain pairs |
| `step4_hierarchy.json` | 329 KB | Meta-graph, PageRank, layer centrality |
| `step5_celltype_circuits.json` | 129 KB | Tissue classification + enrichment |
| `step1_gene_predictions.json` | 134 MB | 975K gene-pair predictions with validation |
| `step2_perturbation_validation.json` | 39 KB | CRISPRi validation results |
| `step3_disease_mapping.json` | 230 KB | Disease gene set enrichment + circuit mapping |

## Citation

```bibtex
@article{kendiukhov2025circuits,
  title={Causal Circuit Tracing Reveals Distinct Computational Architectures
         in Single-Cell Foundation Models: Inhibitory Dominance, Biological
         Coherence, and Cross-Model Convergence},
  author={Kendiukhov, Ihor},
  year={2025},
  note={Preprint}
}

@article{kendiukhov2025sae_atlas,
  title={Sparse Autoencoders Reveal Organized Biological Knowledge but
         Minimal Regulatory Logic in Single-Cell Foundation Models},
  author={Kendiukhov, Ihor},
  journal={Genome Biology},
  year={2025},
  note={Under review}
}
```

## Bioinformatics revision (2026)

In response to reviewer concerns on the first-round *Bioinformatics*
submission, this repository now includes twelve additional experiments
(E1–E12) and one schematic-figure generator (F1) under
[`experiments/revision/`](experiments/revision/). A full mapping from
reviewer points to experiments is in
[`docs/revision_bioinformatics.md`](docs/revision_bioinformatics.md). Highlights:

- **E1** (Replogle RPE1) + **E1b** (Shifrut primary CD8⁺ T cells) extend
  CRISPRi validation to non-malignant / non-immortalized cells.
- **E3** partitions validation pairs into ChIP-seq-supported direct vs
  indirect targets (ENCODE K562 network).
- **E4** replaces the merged-TRRUST enrichment with matched-cell-type
  ENCODE ChIP-seq (2.06× for K562, OR 5.84).
- **E5** recomputes the cross-model effect-size comparison under feature-share
  normalisation, reversing the raw-\|d\| story.
- **E6** repeats circuit tracing with randomly-sampled source features to
  quantify annotation-selection bias.
- **E7** tests whether circuit \|d\| is reducible to marginal gene-gene
  co-expression (R² = 0.010 — it is not).
- **E9**/**E10** add sample-size and per-cell-type stability checks.
- **E11** adds an FDR-controlled threshold derivation alongside the
  magnitude sweep.
- **E12** replaces the reported coherence percentages with explicit
  fold-enrichment over configuration-preserving permutation nulls.

Two new CLI flags have also been added to `src/13_causal_circuit_tracing.py`:

- `--random-feature-seed <int>`: randomly sample source features from the
  annotation-qualified pool (used by **E6**).
- `--ts-tissue <immune|kidney|lung>` and `--ts-cell-type "<label>"`:
  restrict Tabula Sapiens input to a single tissue / cell type (used by
  **E10**).

## Related Repositories

- [bio-sae](https://github.com/Biodyn-AI/bio-sae) -- SAE training pipeline and feature atlases (Phases 1--3)
- [Geneformer Feature Atlas](https://biodyn-ai.github.io/geneformer-atlas/) -- Interactive web atlas
- [scGPT Feature Atlas](https://biodyn-ai.github.io/scgpt-atlas/) -- Interactive web atlas

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE) file for details.
