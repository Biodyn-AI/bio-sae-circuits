"""
E4: Gene-regulatory coherence against ENCODE ChIP-seq TF-target edges.

The paper currently reports TRRUST (merged TF network) 1.12× enrichment.
Reviewer R1 asks for a matched-cell-type ChIP-seq-derived regulatory prior.
ENCODE's 5-cell-line ChIP-seq TF-target edges (1.52M edges, K562 included)
at subproject_53_scgpt_gpl_replication/data/encode/encode_tf_targets_5celllines_edges.tsv
provide this.

Approach:
  - Build a TF-target edge set S_ENCODE = {(TF_symbol, target_symbol)}.
  - For each circuit condition, derive gene-pair candidates from edges by taking
    the Cartesian product of (source_genes × target_genes) for each edge — these
    are the gene-pairs the circuit edge implicitly predicts as interacting.
    Deduplicate.
  - Intersection with ENCODE (restricted to pairs where the source gene is in
    the ENCODE TF vocabulary; targets can be any gene).
  - Fisher's exact test vs a background of all (TF, target) pairs in the ENCODE
    TF vocabulary × ENCODE target vocabulary.
  - Report per condition: n_predicted, n_TF_predicted, n_in_encode, enrichment
    ratio, Fisher p, odds ratio.

Companion output: per-condition table for direct use in revised Table 3 /
Discussion.
"""
from __future__ import annotations

import os

import json
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
ENCODE_PATH = Path(
    os.path.join(os.environ.get("ENCODE_DIR", "../../data/encode"), "encode_tf_targets_5celllines_edges.tsv")
)
EDGES_PATH = REPO / "experiments" / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
OUT_PATH = REPO / "experiments" / "revision_bioinformatics" / "E4_chipseq_coherence" / "chipseq_coherence.json"


def load_encode():
    tfs = set()
    targets = set()
    edges = set()
    with open(ENCODE_PATH) as f:
        header = f.readline().strip().split("\t")
        assert header[:2] == ["TF", "target"], header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tf, tgt = parts[0], parts[1]
            tfs.add(tf)
            targets.add(tgt)
            edges.add((tf, tgt))
    return tfs, targets, edges


def edge_to_gene_pairs(edge, encode_tfs):
    src_genes = edge.get("source_genes") or []
    tgt_genes = edge.get("target_genes") or []
    out = []
    for s in src_genes:
        if s not in encode_tfs:
            continue
        for t in tgt_genes:
            if s == t:
                continue
            out.append((s, t))
    return out


def fisher_exact(a, b, c, d):
    """Right-tail Fisher's exact p-value; falls back to scipy.

    Contingency table:
        in_S   out_S
      in_T   a      b
      out_T  c      d
    """
    try:
        from scipy.stats import fisher_exact as _fe
        odds, p = _fe([[a, b], [c, d]], alternative="greater")
        return float(p), float(odds)
    except Exception:
        # normal approximation via log-odds is crude — require scipy
        return float("nan"), float("nan")


def main():
    print("Loading ENCODE TF-target edges...", flush=True)
    tfs, targets, encode_edges = load_encode()
    print(f"  {len(tfs)} TFs, {len(targets)} targets, {len(encode_edges)} edges", flush=True)

    print("Loading circuit edges...", flush=True)
    with open(EDGES_PATH) as f:
        all_edges = json.load(f)

    N_universe = len(tfs) * len(targets)  # background TF×target pairs
    n_encode = len(encode_edges)

    results = {
        "_universe": {
            "n_tfs": len(tfs),
            "n_targets": len(targets),
            "n_pairs_in_universe": N_universe,
            "n_encode_edges": n_encode,
            "encode_edge_density": n_encode / N_universe,
        }
    }

    for cond, edges in all_edges.items():
        print(f"\n[{cond}]", flush=True)
        predicted = set()
        predicted_tfs = set()
        for e in edges:
            for (s, t) in edge_to_gene_pairs(e, tfs):
                if t in targets:  # restrict to ENCODE target universe
                    predicted.add((s, t))
                    predicted_tfs.add(s)
        n_pred = len(predicted)
        n_hit = len(predicted & encode_edges)
        a = n_hit
        b = n_pred - n_hit
        c = n_encode - n_hit
        d = N_universe - n_encode - b
        p, odds = fisher_exact(a, b, c, d)
        enrichment = (n_hit / n_pred) / (n_encode / N_universe) if n_pred else 0
        print(f"  predicted gene pairs (TF in ENCODE): {n_pred:,}", flush=True)
        print(f"  in ENCODE: {n_hit:,}  enrichment={enrichment:.3f}×  "
              f"Fisher p={p:.3e}  OR={odds:.2f}", flush=True)
        results[cond] = {
            "n_predicted_pairs": n_pred,
            "n_unique_TFs_used": len(predicted_tfs),
            "n_overlap_with_encode": n_hit,
            "enrichment_ratio": enrichment,
            "fisher_exact_p_greater": p,
            "odds_ratio": odds,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
