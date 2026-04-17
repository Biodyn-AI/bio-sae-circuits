"""
E4 refinement: restrict ENCODE TF-target analysis to TFs with K562 ChIP-seq
experiments (from InputsV3). This provides a tighter matched-cell-type prior
without requiring a full BED reparse.
"""
from __future__ import annotations

import os

import json
from pathlib import Path
from collections import defaultdict

REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
ENCODE_DIR = Path(os.environ.get("ENCODE_DIR", "../../data/encode"))
EDGES_TSV = ENCODE_DIR / "encode_tf_targets_5celllines_edges.tsv"
INPUTS_TAB = ENCODE_DIR / "wgEncodeRegTfbsClusteredInputsV3.tab"
CIRCUIT_EDGES = REPO / "experiments" / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
OUT = REPO / "experiments" / "revision_bioinformatics" / "E4_chipseq_coherence" / "k562_restricted.json"


def load_k562_tfs():
    k562 = set()
    with open(INPUTS_TAB) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            tf, cell = parts[2], parts[4]
            if cell == "K562":
                k562.add(tf)
    return k562


def load_encode_filtered(allowed_tfs):
    tfs = set()
    targets = set()
    edges = set()
    with open(EDGES_TSV) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tf, tgt = parts[0], parts[1]
            if tf not in allowed_tfs:
                continue
            tfs.add(tf)
            targets.add(tgt)
            edges.add((tf, tgt))
    return tfs, targets, edges


def fisher(a, b, c, d):
    try:
        from scipy.stats import fisher_exact
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        return float(p), float(odds)
    except Exception:
        return float("nan"), float("nan")


def main():
    k562_tfs = load_k562_tfs()
    print(f"K562 TFs in InputsV3: {len(k562_tfs)}", flush=True)
    tfs, targets, encode_edges = load_encode_filtered(k562_tfs)
    print(f"K562-restricted edge set: {len(tfs)} TFs, {len(targets)} targets, {len(encode_edges)} edges")
    N = len(tfs) * len(targets)
    print(f"Background universe: {N:,} TF×target pairs; K562-restricted encode edges density: {len(encode_edges)/N:.4f}")

    with open(CIRCUIT_EDGES) as f:
        all_edges = json.load(f)

    results = {
        "_note": "K562-restricted ENCODE sub-network: TFs with any K562 ChIP-seq experiment in InputsV3.",
        "n_k562_tfs_in_inputsv3": len(k562_tfs),
        "n_tfs_in_filtered_edge_set": len(tfs),
        "n_targets": len(targets),
        "n_encode_edges": len(encode_edges),
        "universe_density": len(encode_edges) / N,
    }
    for cond, edges in all_edges.items():
        predicted = set()
        predicted_tfs = set()
        for e in edges:
            sg = e.get("source_genes") or []
            tg = e.get("target_genes") or []
            for s in sg:
                if s not in tfs:
                    continue
                for t in tg:
                    if s == t:
                        continue
                    if t in targets:
                        predicted.add((s, t))
                        predicted_tfs.add(s)
        n_pred = len(predicted)
        n_hit = len(predicted & encode_edges)
        a = n_hit
        b = n_pred - n_hit
        c = len(encode_edges) - n_hit
        d = N - len(encode_edges) - b
        p, odds = fisher(a, b, c, d)
        enrich = (n_hit / n_pred) / (len(encode_edges) / N) if n_pred else 0
        print(f"[{cond}] pred={n_pred:,}  hit={n_hit:,}  enrich={enrich:.3f}×  Fisher p={p:.2e}  OR={odds:.2f}")
        results[cond] = {
            "n_predicted_pairs": n_pred,
            "n_unique_TFs_used": len(predicted_tfs),
            "n_overlap": n_hit,
            "enrichment": enrich,
            "fisher_p": p,
            "odds_ratio": odds,
        }

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
