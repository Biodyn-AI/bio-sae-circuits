"""
E7: Are circuit edges reducible to marginal co-expression?

R2-M1 asks whether the feature-feature causal graph encodes anything beyond
marginal gene-gene correlations in the raw expression data. We test three
things on the K562/K562 GF condition:

  (a) Edge-level correlation. For each circuit edge e, compute
        coexp(e) = max_{s in source_genes, t in target_genes} |corr(x_s, x_t)|
      on K562 control cells. Regress |cohens_d| on coexp; report R^2 and
      Spearman rho.

  (b) Directional asymmetry. Marginal correlation is symmetric; circuit
      edges are not. For feature pairs (A,B) where both A->B and B->A exist,
      report the distribution of |d(A->B) - d(B->A)|. If asymmetry is
      common, the circuit encodes directional structure that raw correlation
      cannot.

  (c) Partial correlation of driver gene sets. For a sample of edges, compute
      partial correlation between source and target driver-gene means,
      controlling for all other source features' driver-gene means. Residual
      signal is evidence for structure that is not explained by marginal
      co-expression.

Inputs:
  Replogle h5ad at biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad
  Annotated edges at phase5_knowledge_extraction/step1_annotated_edges.json
Outputs:
  results.json
"""
from __future__ import annotations

import os

import json
import random
from pathlib import Path

import h5py
import numpy as np

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
REPLOGLE = Path(os.environ.get("REPLOGLE_H5AD", "../../data/replogle_concat.h5ad"))
EDGES_PATH = REPO / "experiments" / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
OUT_PATH = REPO / "experiments" / "revision_bioinformatics" / "E7_partial_correlation" / "results.json"

N_CONTROL_CELLS = 500  # K562 non-targeting; enough for stable correlations
MAX_DRIVERS_PER_SIDE = 20  # cap for speed


def load_k562_control_expression(n_cells: int):
    """Return (gene_names, X) for n_cells K562 non-targeting cells.

    Uses the HVG matrix (obsm/X_hvg, 2000 genes) when the var/highly_variable
    mask agrees; falls back to subsetting X by var.gene_name_index if needed.
    """
    with h5py.File(REPLOGLE, "r") as f:
        cl_cats = [b.decode() if isinstance(b, bytes) else b for b in f["obs/cell_line/categories"][:]]
        cl_codes = f["obs/cell_line/codes"][:]
        k562_code = cl_cats.index("k562")
        gene_cats = [b.decode() if isinstance(b, bytes) else b for b in f["obs/gene/categories"][:]]
        nt_code = gene_cats.index("non-targeting")
        gene_codes = f["obs/gene/codes"][:]
        mask = (cl_codes == k562_code) & (gene_codes == nt_code)
        idx = np.where(mask)[0]
        print(f"  K562 non-targeting cells: {len(idx)}", flush=True)
        rng = np.random.default_rng(0)
        if len(idx) > n_cells:
            idx = rng.choice(idx, size=n_cells, replace=False)
            idx.sort()
        var_names = [b.decode() if isinstance(b, bytes) else b for b in f["var/gene_name_index"][:]]
        # Use full X (all 6546 genes) so we can look up any driver gene
        X = f["X"]
        # Chunked read
        mat = np.empty((len(idx), X.shape[1]), dtype=np.float32)
        for i, ii in enumerate(idx):
            mat[i] = X[ii]
    return var_names, mat


def correlations(gene_names: list[str], X: np.ndarray, genes: list[str]):
    """Return a standardized sub-matrix for the requested genes, dropping
    genes not in the vocabulary or with zero variance."""
    name_to_col = {g: i for i, g in enumerate(gene_names)}
    cols = [name_to_col[g] for g in genes if g in name_to_col]
    if not cols:
        return None, []
    sub = X[:, cols]
    mu = sub.mean(0)
    sd = sub.std(0) + 1e-8
    sub = (sub - mu) / sd
    kept = [g for g in genes if g in name_to_col]
    return sub, kept


def edge_coexp(src_sub: np.ndarray, tgt_sub: np.ndarray) -> float:
    """Max |Pearson corr| between any source gene column and any target gene column."""
    if src_sub is None or tgt_sub is None:
        return float("nan")
    n = src_sub.shape[0]
    # correlations via standardized dot product / n
    corr = (src_sub.T @ tgt_sub) / n
    return float(np.abs(corr).max())


def main():
    print("Loading Replogle K562 control cells...", flush=True)
    gene_names, X = load_k562_control_expression(N_CONTROL_CELLS)
    print(f"  Expression matrix: {X.shape}", flush=True)

    print("Loading annotated edges...", flush=True)
    with open(EDGES_PATH) as f:
        all_edges = json.load(f)

    cond = "K562_K562_GF"
    edges = all_edges[cond]
    print(f"  {cond}: {len(edges)} edges", flush=True)

    # (a) Edge-level correlation
    import time
    t0 = time.time()
    coexp_scores = []
    abs_ds = []
    n_valid = 0
    for i, e in enumerate(edges):
        sg = (e.get("source_genes") or [])[:MAX_DRIVERS_PER_SIDE]
        tg = (e.get("target_genes") or [])[:MAX_DRIVERS_PER_SIDE]
        src_sub, _ = correlations(gene_names, X, sg)
        tgt_sub, _ = correlations(gene_names, X, tg)
        c = edge_coexp(src_sub, tgt_sub)
        if np.isfinite(c):
            coexp_scores.append(c)
            abs_ds.append(abs(e["cohens_d"]))
            n_valid += 1
        if (i + 1) % 5000 == 0:
            print(f"    processed {i+1}/{len(edges)} edges in {time.time()-t0:.1f}s", flush=True)
    print(f"  edges with both driver sets in vocab: {n_valid}", flush=True)

    coexp_scores = np.asarray(coexp_scores)
    abs_ds = np.asarray(abs_ds)
    # Pearson R between |d| and coexp
    from numpy import corrcoef
    r = float(corrcoef(coexp_scores, abs_ds)[0, 1]) if n_valid > 1 else float("nan")
    # Spearman
    try:
        from scipy.stats import spearmanr, kendalltau
        sp = spearmanr(coexp_scores, abs_ds)
        sp_r, sp_p = float(sp.correlation), float(sp.pvalue)
    except Exception:
        sp_r, sp_p = float("nan"), float("nan")

    print(f"  |d| vs coexp: Pearson r={r:.4f}  R^2={r**2:.4f}  Spearman rho={sp_r:.4f} (p={sp_p:.2e})", flush=True)

    # (b) Directional asymmetry
    # index by unordered feature pair
    from collections import defaultdict
    fwd = defaultdict(list)  # (A,B) -> list of d values for A->B
    for e in edges:
        key = ((e["source_layer"], e["source_feature"]),
               (e["target_layer"], e["target_feature"]))
        fwd[key].append(e["cohens_d"])
    # find pairs where reverse also exists
    sym_pairs = []
    for (a, b), ds_ab in fwd.items():
        if (b, a) in fwd:
            ds_ba = fwd[(b, a)]
            # use mean per direction if multiple
            d_ab = float(np.mean(ds_ab))
            d_ba = float(np.mean(ds_ba))
            sym_pairs.append((d_ab, d_ba))
    print(f"  reciprocal edge pairs: {len(sym_pairs)}", flush=True)
    if sym_pairs:
        diffs = np.abs(np.array([a - b for a, b in sym_pairs]))
        asym_stats = {
            "n_pairs": len(sym_pairs),
            "mean_abs_diff": float(diffs.mean()),
            "median_abs_diff": float(np.median(diffs)),
            "pct_diff_over_0p5": float((diffs > 0.5).mean()),
            "pct_diff_over_1p0": float((diffs > 1.0).mean()),
        }
    else:
        asym_stats = {"n_pairs": 0}

    out = {
        "condition": cond,
        "n_control_cells": int(X.shape[0]),
        "n_edges_total": len(edges),
        "n_edges_with_coexp": int(n_valid),
        "edge_level": {
            "pearson_r_d_vs_coexp": r,
            "pearson_r_squared": r ** 2 if np.isfinite(r) else None,
            "spearman_rho": sp_r,
            "spearman_p": sp_p,
            "coexp_mean": float(np.mean(coexp_scores)) if n_valid else None,
            "coexp_median": float(np.median(coexp_scores)) if n_valid else None,
            "abs_d_mean": float(np.mean(abs_ds)) if n_valid else None,
        },
        "directional_asymmetry": asym_stats,
        "note": (
            "R^2 reports the variance in |d| explainable by maximum marginal "
            "co-expression of driver genes. Low R^2 is evidence that circuit "
            "edges encode structure beyond bivariate co-expression. "
            "Asymmetry stats quantify direction-dependence that marginal "
            "correlation cannot capture."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
