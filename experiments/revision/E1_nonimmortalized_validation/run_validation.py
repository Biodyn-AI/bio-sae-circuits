"""
E1 + E2: Non-immortalized (RPE1) CRISPRi validation of circuit predictions,
with guide-efficacy filter.

R1-M1 asks for CRISPRi validation on non-immortalized cells from a training-
relevant cell type, and for restriction to guides with statistically
significant target knockdown. Replogle 2022 provides both K562 (immortalized
CML) and RPE1 (near-karyotypically-normal, hTERT-immortalized retinal pigment
epithelial — the standard non-cancer comparison in this screen) in one h5ad.

Approach:
  1. For each targeted gene G in cell line C, pool all G-targeting cells vs
     non-targeting cells and compute:
       - efficacy log2FC and t-test on gene G itself (guide efficacy)
       - response log2FC on all other genes (perturbation response)
  2. Build a per-(C, G) perturbation-response vector.
  3. For each circuit edge e = (source_feature → target_feature) with top
     driver genes (source_genes S, target_genes T), extract predicted gene-
     level effects: for each s in S, if s is a Replogle target gene, predict
     log2FC sign(-cohens_d) and magnitude on each t in T. (Negative d =
     source activation supports target activation, so ablation reduces
     target → knockdown of s should reduce t.)
  4. Compare predicted vs observed on:
       - directional accuracy: sign(predicted) == sign(observed)
       - Spearman correlation between |predicted| and |observed|
  5. Do this for:
       - K562 (baseline, should match the paper's reported 56.4%)
       - K562 with guide-efficacy filter
       - RPE1 (non-malignant comparison)
       - RPE1 with guide-efficacy filter

Outputs: validation_results.json
"""
from __future__ import annotations

import os

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPLOGLE = Path(os.environ.get("REPLOGLE_H5AD", "../../data/replogle_concat.h5ad"))
REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
EDGES_PATH = REPO / "experiments" / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
OUT = REPO / "experiments" / "revision_bioinformatics" / "E1_nonimmortalized_validation" / "validation_results.json"

EFFICACY_LOG2FC = -0.5
EFFICACY_P = 0.05


def welch_t(a: np.ndarray, b: np.ndarray):
    """Return (t, p). Two-sided. Small fast implementation for vectors."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0, 1.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb) + 1e-12
    t = (ma - mb) / se
    # approximate df via Welch-Satterthwaite
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / max(na - 1, 1) + (vb / nb) ** 2 / max(nb - 1, 1))
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(t), df)
    except Exception:
        # normal approx
        from math import erfc, sqrt
        p = erfc(abs(t) / sqrt(2))
    return float(t), float(p)


def build_perturbation_response(cell_line: str, min_cells: int = 30):
    """For each targeted gene G in given cell line, return:
        efficacy[G]    -> (log2fc_self, p_self, n_cells)
        response[G]    -> np.ndarray shape (n_var_genes,) of log2fc per gene
    Pooled across all guides for G.
    """
    print(f"[{cell_line}] loading Replogle...", flush=True)
    with h5py.File(REPLOGLE, "r") as f:
        cl_cats = [b.decode() if isinstance(b, bytes) else b for b in f["obs/cell_line/categories"][:]]
        cl_code = cl_cats.index(cell_line)
        cl_codes = f["obs/cell_line/codes"][:]
        gene_cats = [b.decode() if isinstance(b, bytes) else b for b in f["obs/gene/categories"][:]]
        nt_code = gene_cats.index("non-targeting")
        gene_codes = f["obs/gene/codes"][:]
        var_names = [b.decode() if isinstance(b, bytes) else b for b in f["var/gene_name_index"][:]]

        cl_mask = cl_codes == cl_code
        nt_idx = np.where(cl_mask & (gene_codes == nt_code))[0]
        print(f"  {cell_line} cells: {cl_mask.sum()}, NT cells: {len(nt_idx)}", flush=True)

        # Mean expression on NT cells (reference)
        n_var = f["X"].shape[1]
        # Subsample NT if huge (to keep memory sensible)
        rng = np.random.default_rng(0)
        if len(nt_idx) > 5000:
            nt_sub = np.sort(rng.choice(nt_idx, 5000, replace=False))
        else:
            nt_sub = nt_idx
        # Read NT expression in chunks
        nt_X = f["X"][nt_sub, :]
        nt_mean = np.log2(nt_X.mean(0) + 1e-3)

        # Per-target gene, pool cells
        targeted_genes = [g for i, g in enumerate(gene_cats) if g != "non-targeting"]
        efficacy = {}
        response = {}
        # Build an index of cells per gene_code within this cell line
        gcodes_in_cl = gene_codes[cl_mask]
        rel_cell_idx = np.where(cl_mask)[0]  # absolute row indices of cell-line cells
        per_gene_codes = defaultdict(list)
        for i, gc in enumerate(gcodes_in_cl):
            per_gene_codes[int(gc)].append(int(rel_cell_idx[i]))
        # Process each target
        var_to_col = {v: i for i, v in enumerate(var_names)}
        for g_code, rows in per_gene_codes.items():
            if g_code == nt_code:
                continue
            gname = gene_cats[g_code]
            if len(rows) < min_cells:
                continue
            # subsample if huge
            if len(rows) > 2000:
                rows = rng.choice(rows, 2000, replace=False).tolist()
            rows.sort()
            pert_X = f["X"][rows, :]
            pert_mean_log = np.log2(pert_X.mean(0) + 1e-3)
            resp = pert_mean_log - nt_mean  # log2FC vector (n_var,)
            # self efficacy
            if gname in var_to_col:
                col = var_to_col[gname]
                # per-cell values
                t, p = welch_t(pert_X[:, col], nt_X[:, col])
                # log2FC on the self gene (already in resp[col])
                efficacy[gname] = {
                    "log2fc": float(resp[col]),
                    "t": float(t),
                    "p": float(p),
                    "n_cells": int(len(rows)),
                }
            response[gname] = resp
        print(f"  processed {len(response)} targeted genes in {cell_line}", flush=True)
        return var_names, efficacy, response


def validate(var_names, efficacy, response, all_edges, cond="K562_K562_GF",
             require_efficacy: bool = False, tag: str = ""):
    """Score circuit predictions against observed perturbation response.

    Rule: an edge e with source driver gene s (in perturbation vocab) and
    target driver gene t (in var vocab) predicts sign(-cohens_d) for the
    log2FC of t when s is knocked down (ablating source reduces target
    iff d<0; if we knock down source we should see reduced target, i.e.,
    log2FC < 0). So predicted_sign = sign(cohens_d).
    """
    edges = all_edges[cond]
    var_to_col = {v: i for i, v in enumerate(var_names)}
    # Aggregate one (source_gene, target_gene) -> signed d by the |d|-weighted
    # mean of edges covering this pair. Matches the paper's per-pair protocol
    # rather than an edge-count-weighted one.
    pair_ds: dict[tuple[str, str], list[float]] = defaultdict(list)
    for e in edges:
        s_genes = e.get("source_genes") or []
        t_genes = e.get("target_genes") or []
        d = e["cohens_d"]
        for s in s_genes[:15]:
            for t in t_genes[:15]:
                if s == t:
                    continue
                pair_ds[(s, t)].append(d)
    # Aggregate
    pair_d: dict[tuple[str, str], float] = {}
    for key, ds in pair_ds.items():
        # weighted mean by |d|
        w = np.abs(ds)
        pair_d[key] = float(np.average(ds, weights=w) if w.sum() > 0 else np.mean(ds))

    agree = 0
    total = 0
    preds = []
    obs = []
    targets_used = set()
    for (s, t), d in pair_d.items():
        if s not in response:
            continue
        if require_efficacy:
            eff = efficacy.get(s)
            if eff is None:
                continue
            if not (eff["log2fc"] < EFFICACY_LOG2FC and eff["p"] < EFFICACY_P):
                continue
        if t not in var_to_col:
            continue
        targets_used.add(s)
        obs_val = float(response[s][var_to_col[t]])
        pred_sign = np.sign(d)
        obs_sign = np.sign(obs_val)
        if pred_sign != 0 and obs_sign != 0:
            total += 1
            if pred_sign == obs_sign:
                agree += 1
            preds.append(abs(d))
            obs.append(abs(obs_val))
    dir_acc = (agree / total) if total else 0.0
    # Sign-bias baseline: P(pred<0)*P(obs<0) + P(pred>0)*P(obs>0) over evaluated pairs.
    pred_signs = []
    obs_signs = []
    for (s, t), d in pair_d.items():
        if s not in response or t not in var_to_col:
            continue
        if require_efficacy:
            eff = efficacy.get(s)
            if eff is None or not (eff["log2fc"] < EFFICACY_LOG2FC and eff["p"] < EFFICACY_P):
                continue
        obs_val = float(response[s][var_to_col[t]])
        ps = np.sign(d)
        os_ = np.sign(obs_val)
        if ps != 0 and os_ != 0:
            pred_signs.append(ps)
            obs_signs.append(os_)
    if pred_signs:
        p_neg_pred = float(np.mean([1 if s < 0 else 0 for s in pred_signs]))
        p_neg_obs = float(np.mean([1 if s < 0 else 0 for s in obs_signs]))
        sign_bias_null = p_neg_pred * p_neg_obs + (1 - p_neg_pred) * (1 - p_neg_obs)
    else:
        p_neg_pred = p_neg_obs = sign_bias_null = float("nan")
    try:
        from scipy.stats import spearmanr
        if preds:
            sp = spearmanr(preds, obs)
            sp_r, sp_p = float(sp.correlation), float(sp.pvalue)
        else:
            sp_r, sp_p = float("nan"), float("nan")
    except Exception:
        sp_r, sp_p = float("nan"), float("nan")
    return {
        "tag": tag,
        "condition": cond,
        "require_efficacy": require_efficacy,
        "n_source_genes_used": len(targets_used),
        "n_gene_pairs_evaluated": total,
        "directional_accuracy": dir_acc,
        "sign_bias_null_accuracy": sign_bias_null,
        "excess_over_sign_bias": dir_acc - sign_bias_null if np.isfinite(sign_bias_null) else None,
        "frac_predictions_negative": p_neg_pred,
        "frac_observations_negative": p_neg_obs,
        "spearman_rho_abs_d_abs_logfc": sp_r,
        "spearman_p": sp_p,
    }


def main():
    with open(EDGES_PATH) as f:
        all_edges = json.load(f)

    results = []

    # K562
    var_k562, eff_k562, resp_k562 = build_perturbation_response("k562")
    results.append(validate(var_k562, eff_k562, resp_k562, all_edges,
                            require_efficacy=False, tag="K562_all_guides"))
    results.append(validate(var_k562, eff_k562, resp_k562, all_edges,
                            require_efficacy=True, tag="K562_efficacy_filtered"))

    # RPE1 (non-malignant)
    var_rpe1, eff_rpe1, resp_rpe1 = build_perturbation_response("rpe1")
    results.append(validate(var_rpe1, eff_rpe1, resp_rpe1, all_edges,
                            require_efficacy=False, tag="RPE1_all_guides"))
    results.append(validate(var_rpe1, eff_rpe1, resp_rpe1, all_edges,
                            require_efficacy=True, tag="RPE1_efficacy_filtered"))

    # Report efficacy retention rates
    efficacy_summary = {
        "K562": {
            "n_targets": len(eff_k562),
            "n_passing": sum(1 for e in eff_k562.values()
                             if e["log2fc"] < EFFICACY_LOG2FC and e["p"] < EFFICACY_P),
        },
        "RPE1": {
            "n_targets": len(eff_rpe1),
            "n_passing": sum(1 for e in eff_rpe1.values()
                             if e["log2fc"] < EFFICACY_LOG2FC and e["p"] < EFFICACY_P),
        },
    }
    print("\nEfficacy summary:", efficacy_summary, flush=True)
    print("\nValidation results:")
    for r in results:
        print(f"  [{r['tag']}] n_targets={r['n_source_genes_used']}, "
              f"n_pairs={r['n_gene_pairs_evaluated']}, "
              f"dir_acc={r['directional_accuracy']:.4f}, "
              f"Spearman rho={r['spearman_rho_abs_d_abs_logfc']:.4f} (p={r['spearman_p']:.2e})",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"results": results, "efficacy_summary": efficacy_summary,
                   "efficacy_thresholds": {"log2fc": EFFICACY_LOG2FC, "p": EFFICACY_P}},
                  f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
