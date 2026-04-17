"""
E6 analysis: compare random-feature-subset circuits to annotation-selected.

Metrics (per seed):
  - Aggregate stats: n_edges, mean |d|, inhibitory %, shared-ontology %
  - Same-scale contrast vs the annotation-selected E9 N=50 run on 20 features
    (both are 20 features × 50 cells on L0, differing only in feature selection)

The interesting question: does the 52-68% biological coherence we report on
annotation-qualified features also appear on randomly-sampled features?
If yes, annotation bias is not the driver.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
BASE = REPO / "experiments" / "revision_bioinformatics" / "E6_random_features"
E9_BASE = REPO / "experiments" / "revision_bioinformatics" / "E9_bootstrap_stability"
ANNOT_DIR = REPO / "experiments" / "phase1_k562" / "sae_models"
OUT = BASE / "random_features_analysis.json"


def load_circuit_edges(path):
    """Return list of (source_feature, target_layer, target_feature, d, consistency)."""
    with open(path) as f:
        d = json.load(f)
    out = []
    for feat in d["features"]:
        fi = feat["source_feature_idx"]
        for dl_str, effects in feat["downstream_effects"].items():
            dl = int(dl_str)
            for e in effects["top_effects"]:
                out.append((fi, dl, e["target_feature_idx"],
                           float(e["cohens_d"]), float(e["consistency"])))
    return out


def load_layer_feature_annotations(layer):
    """Return {feature_idx: frozenset(ontology::term)}."""
    p = ANNOT_DIR / f"layer{layer:02d}_x4_k32" / "feature_annotations.json"
    with open(p) as f:
        d = json.load(f)
    fa = d.get("feature_annotations", {})
    out = {}
    for fid_str, records in fa.items():
        if not isinstance(records, list):
            continue
        terms = set()
        for r in records:
            if not isinstance(r, dict):
                continue
            t = r.get("term")
            o = r.get("ontology")
            if t:
                terms.add(f"{o}::{t}")
        out[int(fid_str)] = frozenset(terms)
    return out


def shared_ontology_pct(edges, ann_L0, ann_by_layer):
    """Fraction of edges with >=1 shared term between source and target."""
    both = 0
    shared = 0
    for (fi, dl, tf, d, c) in edges:
        src = ann_L0.get(fi, frozenset())
        tgt = ann_by_layer.get(dl, {}).get(tf, frozenset())
        if src and tgt:
            both += 1
            if src & tgt:
                shared += 1
    return (shared / both if both else None, shared, both)


def stats(edges, ann_L0, ann_by_layer):
    ds = [e[3] for e in edges]
    if not ds:
        return {}
    n_src = len({e[0] for e in edges})
    n_tgt = len({(e[1], e[2]) for e in edges})
    inh = sum(1 for d in ds if d < 0) / len(ds)
    abs_ds = [abs(d) for d in ds]
    frac, n_shared, n_both = shared_ontology_pct(edges, ann_L0, ann_by_layer)
    return {
        "n_edges": len(edges),
        "n_source_features": n_src,
        "n_target_features": n_tgt,
        "mean_abs_d": float(np.mean(abs_ds)),
        "median_abs_d": float(np.median(abs_ds)),
        "frac_abs_d_gt_1": float(np.mean([1 if x > 1.0 else 0 for x in abs_ds])),
        "inhibitory_frac": inh,
        "shared_ontology_frac": frac,
        "n_shared": n_shared,
        "n_both_annotated": n_both,
    }


def main():
    # Preload annotations across all 18 layers (for target features) and L0 (for source)
    ann_by_layer = {L: load_layer_feature_annotations(L) for L in range(18)}
    ann_L0 = ann_by_layer[0]

    out = {"by_run": {}}

    # E9 annotation-selected N=50 run
    e9_path = E9_BASE / "N50" / "circuit_L00_features.json"
    if e9_path.exists():
        edges = load_circuit_edges(e9_path)
        out["by_run"]["annotation_selected_N50"] = stats(edges, ann_L0, ann_by_layer)
        out["by_run"]["annotation_selected_N50"]["source_feature_indices"] = sorted(
            {e[0] for e in edges}
        )

    # Random seed runs
    for seed_dir in sorted(BASE.glob("seed*")):
        p = seed_dir / "circuit_L00_features.json"
        if not p.exists():
            continue
        edges = load_circuit_edges(p)
        out["by_run"][seed_dir.name] = stats(edges, ann_L0, ann_by_layer)
        out["by_run"][seed_dir.name]["source_feature_indices"] = sorted(
            {e[0] for e in edges}
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT}\n")
    for name, d in out["by_run"].items():
        print(f"[{name}]")
        for k in ["n_edges", "n_source_features", "n_target_features", "mean_abs_d",
                  "inhibitory_frac", "shared_ontology_frac"]:
            v = d.get(k)
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
