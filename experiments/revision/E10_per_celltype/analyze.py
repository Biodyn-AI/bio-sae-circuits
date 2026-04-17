"""
E10 analysis: compare per-cell-type TS circuits against each other and against
the stratified TS/Multi (GF) run from the original paper.

Outputs pairwise edge Jaccard and Pearson r(d).
"""
import json
from pathlib import Path
from itertools import combinations
import numpy as np

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
BASE = REPO / "experiments" / "revision_bioinformatics" / "E10_per_celltype"
TS_STRATIFIED = REPO / "experiments" / "phase3_multitissue" / "circuit_tracing_ts_cells" / "circuit_L00_features.json"
OUT = BASE / "per_celltype_stability.json"


def load_edges(path):
    with open(path) as f:
        d = json.load(f)
    out = []
    for feat in d.get("features", []):
        fi = feat["source_feature_idx"]
        for dl_str, effects in feat.get("downstream_effects", {}).items():
            dl = int(dl_str)
            for e in effects.get("top_effects", []):
                out.append((fi, dl, e["target_feature_idx"],
                            float(e["cohens_d"]), float(e["consistency"])))
    return out


def edge_dict(edges):
    return {(fi, dl, tf): (cd, cs) for (fi, dl, tf, cd, cs) in edges}


def compare_pair(d1, d2):
    s1 = set(d1.keys())
    s2 = set(d2.keys())
    inter = s1 & s2
    union = s1 | s2
    if not inter:
        return {"jaccard": 0.0, "pearson_r": None, "sign_agreement": None,
                "n1": len(s1), "n2": len(s2), "intersection": 0, "union": len(union)}
    v1 = np.array([d1[k][0] for k in inter])
    v2 = np.array([d2[k][0] for k in inter])
    r = float(np.corrcoef(v1, v2)[0, 1]) if len(inter) > 1 else None
    sa = float((np.sign(v1) == np.sign(v2)).mean())
    return {
        "n1": len(s1),
        "n2": len(s2),
        "intersection": len(inter),
        "union": len(union),
        "jaccard": len(inter) / len(union),
        "pearson_r": r,
        "sign_agreement": sa,
    }


def main():
    datasets = {}
    # Per-cell-type runs
    for d in sorted(BASE.iterdir()):
        if not d.is_dir():
            continue
        p = d / "circuit_L00_features.json"
        if p.exists():
            datasets[d.name] = edge_dict(load_edges(p))
            print(f"Loaded {d.name}: {len(datasets[d.name])} edges")

    # Stratified TS/Multi
    if TS_STRATIFIED.exists():
        datasets["stratified_TS_multi"] = edge_dict(load_edges(TS_STRATIFIED))
        print(f"Loaded stratified_TS_multi: {len(datasets['stratified_TS_multi'])} edges")

    out = {"counts": {n: len(d) for n, d in datasets.items()}, "pairwise": {}}
    for (a, b) in combinations(datasets.keys(), 2):
        out["pairwise"][f"{a}__vs__{b}"] = compare_pair(datasets[a], datasets[b])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}\n")
    for k, v in out["pairwise"].items():
        print(f"{k}")
        for kk, vv in v.items():
            if isinstance(vv, float):
                print(f"  {kk}: {vv:.4f}")
            else:
                print(f"  {kk}: {vv}")


if __name__ == "__main__":
    main()
