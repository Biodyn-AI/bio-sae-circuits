"""
E3: Partition validation gene-pairs into DIRECT (source TF binds target in
matched-cell-type ChIP-seq) vs INDIRECT. Recompute directional accuracy +
sign-bias baseline + Spearman on each partition.

Geneformer 2023 showed in-silico deletion effects were stronger for
ChIP-seq-supported direct targets. We test the same here with the ablation-
derived circuits.
"""
from __future__ import annotations

import os

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
REPLOGLE = Path(os.environ.get("REPLOGLE_H5AD", "../../data/replogle_concat.h5ad"))
EDGES_PATH = REPO / "experiments" / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
ENCODE_DIR = Path(os.environ.get("ENCODE_DIR", "../../data/encode"))
OUT = REPO / "experiments" / "revision_bioinformatics" / "E3_direct_indirect" / "results.json"


def load_k562_tfs():
    k562 = set()
    with open(ENCODE_DIR / "wgEncodeRegTfbsClusteredInputsV3.tab") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            if parts[4] == "K562":
                k562.add(parts[2])
    return k562


def load_encode_k562_edges(k562_tfs):
    edges = set()
    with open(ENCODE_DIR / "encode_tf_targets_5celllines_edges.tsv") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] in k562_tfs:
                edges.add((parts[0], parts[1]))
    return edges


def build_response(cell_line: str, min_cells: int = 30, rng_seed: int = 0):
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
        print(f"  {cell_line}: {cl_mask.sum()} cells; {len(nt_idx)} NT", flush=True)
        rng = np.random.default_rng(rng_seed)
        if len(nt_idx) > 5000:
            nt_idx = np.sort(rng.choice(nt_idx, 5000, replace=False))
        nt_X = f["X"][nt_idx, :]
        nt_mean = np.log2(nt_X.mean(0) + 1e-3)

        gcodes_in_cl = gene_codes[cl_mask]
        rel_cell_idx = np.where(cl_mask)[0]
        per_gene_codes = defaultdict(list)
        for i, gc in enumerate(gcodes_in_cl):
            per_gene_codes[int(gc)].append(int(rel_cell_idx[i]))

        response = {}
        for g_code, rows in per_gene_codes.items():
            if g_code == nt_code or len(rows) < min_cells:
                continue
            if len(rows) > 2000:
                rows = rng.choice(rows, 2000, replace=False).tolist()
            rows.sort()
            pert_X = f["X"][rows, :]
            response[gene_cats[g_code]] = np.log2(pert_X.mean(0) + 1e-3) - nt_mean
        print(f"  built response for {len(response)} target genes", flush=True)
        return var_names, response


def evaluate(var_names, response, edges_condition, direct_edges, tag):
    var_to_col = {v: i for i, v in enumerate(var_names)}
    pair_ds = defaultdict(list)
    for e in edges_condition:
        s_genes = e.get("source_genes") or []
        t_genes = e.get("target_genes") or []
        d = e["cohens_d"]
        for s in s_genes[:15]:
            for t in t_genes[:15]:
                if s == t:
                    continue
                pair_ds[(s, t)].append(d)

    # Split into direct/indirect
    buckets = {"direct": [], "indirect": []}
    for (s, t), ds in pair_ds.items():
        if s not in response or t not in var_to_col:
            continue
        obs_val = float(response[s][var_to_col[t]])
        w = np.abs(ds)
        d_agg = float(np.average(ds, weights=w) if w.sum() > 0 else np.mean(ds))
        bucket = "direct" if (s, t) in direct_edges else "indirect"
        buckets[bucket].append((d_agg, obs_val))

    results = {}
    for b, pairs in buckets.items():
        if not pairs:
            results[b] = {"n": 0}
            continue
        pred = np.array([p[0] for p in pairs])
        obs = np.array([p[1] for p in pairs])
        pred_sign = np.sign(pred)
        obs_sign = np.sign(obs)
        valid = (pred_sign != 0) & (obs_sign != 0)
        if not valid.any():
            results[b] = {"n": 0}
            continue
        ps = pred_sign[valid]
        os_ = obs_sign[valid]
        agree = float((ps == os_).mean())
        p_neg_p = float((ps < 0).mean())
        p_neg_o = float((os_ < 0).mean())
        bias_null = p_neg_p * p_neg_o + (1 - p_neg_p) * (1 - p_neg_o)
        try:
            from scipy.stats import spearmanr
            sp = spearmanr(np.abs(pred[valid]), np.abs(obs[valid]))
            sp_r, sp_p = float(sp.correlation), float(sp.pvalue)
        except Exception:
            sp_r, sp_p = float("nan"), float("nan")
        results[b] = {
            "n": int(valid.sum()),
            "directional_accuracy": agree,
            "sign_bias_null": bias_null,
            "excess_over_bias": agree - bias_null,
            "frac_predictions_negative": p_neg_p,
            "frac_observations_negative": p_neg_o,
            "spearman_rho": sp_r,
            "spearman_p": sp_p,
        }
    print(f"[{tag}] direct n={results['direct'].get('n',0)}  "
          f"dir_acc={results['direct'].get('directional_accuracy',0):.4f}  "
          f"bias={results['direct'].get('sign_bias_null',0):.4f}  "
          f"excess={results['direct'].get('excess_over_bias',0):.4f}  "
          f"rho={results['direct'].get('spearman_rho',0):.4f}", flush=True)
    print(f"[{tag}] indirect n={results['indirect'].get('n',0)}  "
          f"dir_acc={results['indirect'].get('directional_accuracy',0):.4f}  "
          f"bias={results['indirect'].get('sign_bias_null',0):.4f}  "
          f"excess={results['indirect'].get('excess_over_bias',0):.4f}  "
          f"rho={results['indirect'].get('spearman_rho',0):.4f}", flush=True)
    return results


def main():
    print("Loading K562 TFs + ENCODE K562 edges...", flush=True)
    k562_tfs = load_k562_tfs()
    direct_edges = load_encode_k562_edges(k562_tfs)
    print(f"  {len(k562_tfs)} K562 TFs; {len(direct_edges):,} K562-restricted edges", flush=True)

    print("Loading annotated circuit edges...", flush=True)
    with open(EDGES_PATH) as f:
        all_edges = json.load(f)
    cond = "K562_K562_GF"
    edges = all_edges[cond]

    out = {
        "n_k562_tfs": len(k562_tfs),
        "n_encode_k562_edges": len(direct_edges),
        "condition": cond,
    }

    print("\nBuilding K562 perturbation response...", flush=True)
    var_k562, resp_k562 = build_response("k562")
    out["K562"] = evaluate(var_k562, resp_k562, edges, direct_edges, "K562")

    print("\nBuilding RPE1 perturbation response...", flush=True)
    var_rpe1, resp_rpe1 = build_response("rpe1")
    out["RPE1"] = evaluate(var_rpe1, resp_rpe1, edges, direct_edges, "RPE1")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
