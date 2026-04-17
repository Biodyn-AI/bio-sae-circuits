"""
E5: Input-size-normalized cross-model comparison (R1-M5).

scGPT has 2048 features/layer; Geneformer has 4608. Each feature carries a
larger fraction of the total representation in scGPT, which may contribute to
its higher mean |d|. We test whether the "stronger scGPT effects" story
survives input-size normalization.

Two normalizations:
  1. d_per_feature_share: d * (n_features / n_ref)
     where n_ref is the max feature count (4608). Puts both on a common
     "per-feature-share" scale.
  2. d_per_input: scGPT's effective input is 1200 padded positions; Geneformer
     uses 2048 positions of rank-encoded genes. d_per_input = d * (input_size /
     input_size_ref). Per-position-share.

We compute mean |d| under each normalization for every condition. A third
analysis: for gene pairs (s, t) that appear in BOTH Geneformer circuit edges
and scGPT circuit edges, compute paired d vs d-normalized to test if the
stronger-scGPT pattern holds per-pair.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
EDGES_PATH = REPO / "experiments" / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
OUT = REPO / "experiments" / "revision_bioinformatics" / "E5_input_size_normalization" / "results.json"

# Features-per-layer per condition
N_FEATS = {
    "K562_K562_GF":   4608,
    "K562_Multi_GF":  4608,
    "TS_Multi_GF":    4608,
    "scGPT_TS_Multi": 2048,
}
# Input-position window per model (tokens per cell)
INPUT_SIZE = {
    "K562_K562_GF":   2048,   # Geneformer V2-316M max is 4096 but extraction used 2048 positions
    "K562_Multi_GF":  2048,
    "TS_Multi_GF":    2048,
    "scGPT_TS_Multi": 1200,   # scGPT padded to 1200 positions (paper Methods)
}
N_REF = max(N_FEATS.values())      # 4608
INPUT_REF = max(INPUT_SIZE.values())  # 2048


def summary(edges, scale):
    ds = np.abs([e["cohens_d"] * scale for e in edges])
    return {
        "n_edges": len(edges),
        "mean_abs_d": float(ds.mean()),
        "median_abs_d": float(np.median(ds)),
        "frac_gt_1": float((ds > 1.0).mean()),
        "frac_gt_2": float((ds > 2.0).mean()),
    }


def main():
    with open(EDGES_PATH) as f:
        all_edges = json.load(f)

    out = {
        "normalizations": {
            "feature_share": "d * (n_features / n_ref=4608)",
            "input_share":   "d * (input_size / input_ref=2048)",
        },
        "per_condition": {},
    }

    for cond, edges in all_edges.items():
        out["per_condition"][cond] = {
            "raw":           summary(edges, 1.0),
            "feature_share": summary(edges, N_FEATS[cond] / N_REF),
            "input_share":   summary(edges, INPUT_SIZE[cond] / INPUT_REF),
            "scaling_factor_feature_share": N_FEATS[cond] / N_REF,
            "scaling_factor_input_share":   INPUT_SIZE[cond] / INPUT_REF,
        }

    # Paired gene-pair comparison between K562/K562 GF and scGPT TS
    # Build (source_gene, target_gene) → list(d) for each condition
    def pair_table(edges):
        t = defaultdict(list)
        for e in edges:
            for s in (e.get("source_genes") or [])[:10]:
                for tgt in (e.get("target_genes") or [])[:10]:
                    if s != tgt:
                        t[(s, tgt)].append(e["cohens_d"])
        return {k: float(np.mean(v)) for k, v in t.items()}

    gf = pair_table(all_edges["K562_K562_GF"])
    sg = pair_table(all_edges["scGPT_TS_Multi"])
    common = set(gf.keys()) & set(sg.keys())
    if common:
        gf_v = np.array([gf[k] for k in common])
        sg_v = np.array([sg[k] for k in common])
        gf_norm = gf_v * (N_FEATS["K562_K562_GF"] / N_REF)
        sg_norm = sg_v * (N_FEATS["scGPT_TS_Multi"] / N_REF)
        out["paired_gene_pairs"] = {
            "n_pairs": len(common),
            "raw_mean_abs_gf": float(np.abs(gf_v).mean()),
            "raw_mean_abs_scgpt": float(np.abs(sg_v).mean()),
            "raw_ratio_scgpt_over_gf": float(np.abs(sg_v).mean() / np.abs(gf_v).mean()),
            "feature_share_norm_mean_abs_gf": float(np.abs(gf_norm).mean()),
            "feature_share_norm_mean_abs_scgpt": float(np.abs(sg_norm).mean()),
            "feature_share_norm_ratio_scgpt_over_gf": float(np.abs(sg_norm).mean() / np.abs(gf_norm).mean()),
            "sign_agreement": float((np.sign(gf_v) == np.sign(sg_v)).mean()),
            "pearson_r_d": float(np.corrcoef(gf_v, sg_v)[0, 1]) if len(common) > 1 else None,
        }
    else:
        out["paired_gene_pairs"] = {"n_pairs": 0}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT}")
    print("\nSummary:")
    for cond, data in out["per_condition"].items():
        print(f"  {cond}: raw mean|d|={data['raw']['mean_abs_d']:.3f}  "
              f"feature-share mean|d|={data['feature_share']['mean_abs_d']:.3f}  "
              f"input-share mean|d|={data['input_share']['mean_abs_d']:.3f}")
    if out["paired_gene_pairs"]["n_pairs"]:
        pp = out["paired_gene_pairs"]
        print(f"\nPaired gene pairs (N={pp['n_pairs']:,}):")
        print(f"  raw ratio scGPT/GF = {pp['raw_ratio_scgpt_over_gf']:.3f}")
        print(f"  feature-share ratio scGPT/GF = {pp['feature_share_norm_ratio_scgpt_over_gf']:.3f}")
        print(f"  sign agreement = {pp['sign_agreement']:.3f}; Pearson r = {pp['pearson_r_d']:.3f}")


if __name__ == "__main__":
    main()
