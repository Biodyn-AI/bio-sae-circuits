"""
E9 analysis: compare circuit graphs at N=50, N=100 vs existing N=200.

For each (source_feature, target_layer, target_feature) triple common to
multiple N values, compute:
  - Pearson correlation of Cohen's d across N
  - Fraction of N=200 edges recovered at each N' (|d|>0.5 threshold)
  - Hub identity Jaccard on top-50 out-degree

Inputs: circuit_L00_features.json in each N/ directory.
"""
import json
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
BASE = REPO / "experiments" / "revision_bioinformatics" / "E9_bootstrap_stability"
N_DIRS = {50: BASE / "N50", 100: BASE / "N100"}
N200_PATH = REPO / "experiments" / "phase1_k562" / "circuit_tracing" / "circuit_L00_features.json"
OUT = BASE / "stability_analysis.json"

THRESHOLD = 0.5


def load_edges(path):
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


def edge_dict(edges, fi_filter=None):
    d = {}
    for (fi, dl, tf, cd, c) in edges:
        if fi_filter is not None and fi not in fi_filter:
            continue
        d[(fi, dl, tf)] = (cd, c)
    return d


def main():
    n200 = load_edges(N200_PATH)
    print(f"N=200 edges (all 30 features): {len(n200)}")

    per_n = {200: n200}
    for n, d in N_DIRS.items():
        p = d / "circuit_L00_features.json"
        if p.exists():
            per_n[n] = load_edges(p)
            print(f"N={n} edges: {len(per_n[n])}")
        else:
            print(f"N={n}: file not yet present at {p}")

    # Find the feature set common to all N's (subset of top-30)
    feat_sets = {n: set(e[0] for e in edges) for n, edges in per_n.items()}
    common_features = set.intersection(*feat_sets.values())
    print(f"Common source features: {len(common_features)}")

    # Restrict to common features and compare
    restricted = {n: edge_dict(edges, fi_filter=common_features) for n, edges in per_n.items()}

    results = {"threshold": THRESHOLD, "n_common_source_features": len(common_features)}
    for n, edges in restricted.items():
        results[f"N{n}"] = {
            "n_edges_passing": sum(1 for _, (cd, _c) in edges.items() if abs(cd) >= THRESHOLD),
            "n_edges_all": len(edges),
        }

    # Compute cross-N stability metrics
    pairwise = {}
    for n1 in sorted(restricted.keys()):
        for n2 in sorted(restricted.keys()):
            if n1 >= n2:
                continue
            d1, d2 = restricted[n1], restricted[n2]
            keys = set(d1.keys()) & set(d2.keys())
            if not keys:
                continue
            v1 = np.array([d1[k][0] for k in keys])
            v2 = np.array([d2[k][0] for k in keys])
            # correlation of d
            if len(keys) > 1:
                r = float(np.corrcoef(v1, v2)[0, 1])
            else:
                r = None
            # sign agreement
            sign_agree = float((np.sign(v1) == np.sign(v2)).mean()) if len(keys) else None
            # edge preservation at |d|>THRESHOLD
            in_both = sum(1 for k in keys if abs(d1[k][0]) >= THRESHOLD and abs(d2[k][0]) >= THRESHOLD)
            in_1 = sum(1 for k in keys if abs(d1[k][0]) >= THRESHOLD)
            in_2 = sum(1 for k in keys if abs(d2[k][0]) >= THRESHOLD)
            preserved = {
                f"|d|>={THRESHOLD} at N{n1}": in_1,
                f"|d|>={THRESHOLD} at N{n2}": in_2,
                f"|d|>={THRESHOLD} at both": in_both,
                "jaccard_passing": in_both / (in_1 + in_2 - in_both) if (in_1 + in_2 - in_both) else None,
            }
            # Full edge-set comparison (above the filter)
            set1 = set(d1.keys())
            set2 = set(d2.keys())
            intersection = set1 & set2
            union = set1 | set2
            pairwise[f"N{n1}_vs_N{n2}"] = {
                "n_edges_{0}".format(n1): len(set1),
                "n_edges_{0}".format(n2): len(set2),
                "n_edges_intersection": len(intersection),
                "n_edges_union": len(union),
                "edge_jaccard_full": len(intersection) / len(union) if union else None,
                "recall_of_higher_N_at_lower_N": len(intersection) / len(set2) if set2 else None,
                "precision_of_lower_N_on_higher_N": len(intersection) / len(set1) if set1 else None,
                "n_common_edges": len(keys),
                "pearson_r_d_on_common": r,
                "sign_agreement_on_common": sign_agree,
                **preserved,
            }
    results["pairwise"] = pairwise

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}")
    for k, v in pairwise.items():
        print(f"\n{k}")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")


if __name__ == "__main__":
    main()
