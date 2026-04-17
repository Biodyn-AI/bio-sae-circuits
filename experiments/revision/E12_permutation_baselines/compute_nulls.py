"""
E12: Permutation-based chance baselines for coherence metrics.

For each of the four conditions (K562/K562 GF, K562/Multi GF, TS/Multi GF,
scGPT TS/Multi), compute observed and null distributions for:
  - Shared-ontology % across edges (main text "53% biological coherence").
  - Inhibitory % (sign-randomization null).
  - Source→target layer-direction bias (sanity check).

Cross-model consensus enrichment already has a permutation p-value in
phase5/step2_consensus_graph.json (10.65×, p < 0.001 over 1000 perms).

Inputs:
  - Annotated edges per condition: phase5_knowledge_extraction/step1_annotated_edges.json
  - Per-feature annotation sets from feature_annotations.json across SAE model dirs:
      phase1_k562/sae_models/layer{LL}_x4_k32/feature_annotations.json  (K562-only GF)
      phase3_multitissue/sae_models/layer{LL}_x4_k32/feature_annotations.json  (multi-tissue GF, L∈{00,05,11,17})
      scgpt_atlas/sae_models/layer{LL}_x4_k32/feature_annotations.json  (scGPT TS)
Outputs:
  - nulls.json  (observed + null mean/SD + empirical p-value for each metric × condition)
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
EXPS = REPO / "experiments"
EDGES_PATH = EXPS / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
OUT_PATH = EXPS / "revision_bioinformatics" / "E12_permutation_baselines" / "nulls.json"

# Per-condition SAE feature-annotation lookup:
#   (layer_int) -> path to feature_annotations.json
CONDITION_SAE_ROOTS = {
    "K562_K562_GF": EXPS / "phase1_k562" / "sae_models",         # all 18 layers
    "K562_Multi_GF": EXPS / "phase3_multitissue" / "sae_models", # layers 00 05 11 17
    "TS_Multi_GF":   EXPS / "phase3_multitissue" / "sae_models", # same SAEs
    "scGPT_TS_Multi": EXPS / "scgpt_atlas" / "sae_models",       # all 12 layers
}


def load_feature_annotations(sae_root: Path, layer: int) -> dict[int, frozenset]:
    """Return {feature_idx: frozenset(term strings)} for a single SAE layer."""
    path = sae_root / f"layer{layer:02d}_x4_k32" / "feature_annotations.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    fa = d.get("feature_annotations", {})
    out: dict[int, frozenset] = {}
    for fid_str, records in fa.items():
        if not isinstance(records, list):
            continue
        terms = set()
        for r in records:
            if not isinstance(r, dict):
                continue
            term = r.get("term")
            onto = r.get("ontology")
            if term:
                terms.add(f"{onto}::{term}")
        out[int(fid_str)] = frozenset(terms)
    return out


def build_feature_annotation_table(condition: str) -> dict[tuple[int, int], frozenset]:
    """Return {(layer, feature_idx): terms} covering all layers used by the condition."""
    root = CONDITION_SAE_ROOTS[condition]
    # infer available layers from directory
    layers = []
    for p in root.glob("layer*_x4_k32"):
        try:
            layers.append(int(p.name.split("_")[0][5:]))
        except Exception:
            pass
    table: dict[tuple[int, int], frozenset] = {}
    for L in sorted(layers):
        by_fid = load_feature_annotations(root, L)
        for fid, terms in by_fid.items():
            table[(L, fid)] = terms
    return table


def observed_shared_pct(edges, ann_table):
    """Fraction of edges whose source and target feature share at least one term.

    Matches the paper's coherence definition (line 778 of sae_paper_v4.tex).
    Denominator: edges with at least one term on each side.
    """
    both_annotated = 0
    shared = 0
    for e in edges:
        src = ann_table.get((e["source_layer"], e["source_feature"]), frozenset())
        tgt = ann_table.get((e["target_layer"], e["target_feature"]), frozenset())
        if src and tgt:
            both_annotated += 1
            if src & tgt:
                shared += 1
    if both_annotated == 0:
        return 0.0, 0, 0
    return shared / both_annotated, shared, both_annotated


def null_shared_pct(edges, ann_table, n_perm: int, seed: int = 0):
    """Configuration-preserving null: for each (layer, feature) entry in the
    annotation table, reassign its term set to another entry's at the same
    layer (i.e., shuffle annotations among features of the same layer).

    This preserves: per-layer distribution of term-set sizes, per-layer term
    pool. Breaks: the real feature→term pairing.
    """
    rng = random.Random(seed)
    # Group feature-indices by layer
    by_layer: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (L, fid) in ann_table.keys():
        by_layer[L].append((L, fid))
    null_fracs = []
    for i in range(n_perm):
        # Permute term-set assignment within each layer
        perm_table = {}
        for L, keys in by_layer.items():
            terms = [ann_table[k] for k in keys]
            rng.shuffle(terms)
            for k, t in zip(keys, terms):
                perm_table[k] = t
        both = 0
        shared = 0
        for e in edges:
            src = perm_table.get((e["source_layer"], e["source_feature"]), frozenset())
            tgt = perm_table.get((e["target_layer"], e["target_feature"]), frozenset())
            if src and tgt:
                both += 1
                if src & tgt:
                    shared += 1
        null_fracs.append(shared / both if both else 0.0)
    return null_fracs


def observed_inhibitory_pct(edges):
    neg = sum(1 for e in edges if e["cohens_d"] < 0)
    return neg / len(edges), neg, len(edges)


def binomial_two_sided_p(n_success: int, n: int, p0: float = 0.5):
    """Exact two-sided binomial p-value via scipy if present, else normal approx."""
    try:
        from scipy.stats import binomtest
        return binomtest(n_success, n, p0).pvalue
    except Exception:
        import math
        mu = n * p0
        sd = math.sqrt(n * p0 * (1 - p0))
        z = (n_success - mu) / sd
        # two-sided normal approx
        from math import erfc, sqrt
        return erfc(abs(z) / sqrt(2))


def empirical_p(observed: float, nulls: list[float]) -> float:
    """Two-sided empirical p: fraction of null draws at least as extreme."""
    m = sum(1 for x in nulls if abs(x - sum(nulls) / len(nulls))
            >= abs(observed - sum(nulls) / len(nulls)))
    return m / len(nulls)


def main(n_perm: int):
    print(f"Loading edges from {EDGES_PATH}", flush=True)
    with open(EDGES_PATH) as f:
        all_edges = json.load(f)

    results = {}
    for cond in ["K562_K562_GF", "K562_Multi_GF", "TS_Multi_GF", "scGPT_TS_Multi"]:
        edges = all_edges[cond]
        print(f"\n[{cond}] {len(edges)} edges, loading SAE annotations...", flush=True)
        ann = build_feature_annotation_table(cond)
        print(f"    {len(ann)} (layer, feature) annotated entries", flush=True)

        # 1. Shared ontology
        obs_frac, obs_shared, obs_both = observed_shared_pct(edges, ann)
        print(f"    shared% observed: {obs_frac:.4f} ({obs_shared}/{obs_both})", flush=True)
        print(f"    computing {n_perm} permutations...", flush=True)
        nulls = null_shared_pct(edges, ann, n_perm=n_perm)
        null_mean = sum(nulls) / len(nulls)
        null_sd = (sum((x - null_mean) ** 2 for x in nulls) / len(nulls)) ** 0.5
        p_emp = empirical_p(obs_frac, nulls)
        print(f"    null mean ± sd: {null_mean:.4f} ± {null_sd:.4f}  p≈{p_emp:.4f}", flush=True)

        # 2. Inhibitory %
        inh_frac, neg, total = observed_inhibitory_pct(edges)
        p_binom = binomial_two_sided_p(neg, total, 0.5)
        print(f"    inhibitory% observed: {inh_frac:.4f} ({neg}/{total}) binom p={p_binom:.3e}", flush=True)

        results[cond] = {
            "shared_ontology": {
                "observed": obs_frac,
                "n_shared": obs_shared,
                "n_both_annotated": obs_both,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "p_empirical": p_emp,
                "n_permutations": n_perm,
                "fold_enrichment_over_null": obs_frac / null_mean if null_mean > 0 else None,
            },
            "inhibitory": {
                "observed": inh_frac,
                "n_negative": neg,
                "n_total": total,
                "p_binomial_two_sided": p_binom,
                "excess_over_50pct": inh_frac - 0.5,
            },
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()
    main(args.n_perm)
