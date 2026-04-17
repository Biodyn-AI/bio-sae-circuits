"""
E11: Threshold sensitivity sweep on |d|.

Recompute the headline metrics at |d| ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} for each
of the four conditions. Uses step1_annotated_edges.json (labeled edges with
cohens_d and source/target domain labels).

Metrics swept:
  - n_edges, n_source_features, n_target_features, target_coverage
  - mean |d|, median |d|, fraction(|d|>1), fraction(|d|>2)
  - inhibitory %
  - shared_ontology % (via source_label == target_label OR intersection of
    source_genes and target_genes non-empty; we use the domain-label match that
    the paper uses elsewhere and also report the gene-overlap version)

Outputs:
  threshold_sweep.json  — per-condition × threshold metrics.
"""
from __future__ import annotations

import os

import json
from pathlib import Path
from statistics import mean, median

import sys
REPO = Path(os.environ.get("BIO_SAE_ROOT", "../.."))
sys.path.insert(0, str(REPO / "experiments" / "revision_bioinformatics" / "E12_permutation_baselines"))
from compute_nulls import build_feature_annotation_table  # noqa: E402

EDGES_PATH = REPO / "experiments" / "phase5_knowledge_extraction" / "step1_annotated_edges.json"
OUT_PATH = REPO / "experiments" / "revision_bioinformatics" / "E11_threshold_sweep" / "threshold_sweep.json"

THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def metrics(edges, ann_table):
    if not edges:
        return dict(n_edges=0)
    ds = [e["cohens_d"] for e in edges]
    abs_ds = [abs(x) for x in ds]
    src_feats = {(e["source_layer"], e["source_feature"]) for e in edges}
    tgt_feats = {(e["target_layer"], e["target_feature"]) for e in edges}
    inhib = sum(1 for x in ds if x < 0) / len(ds)

    # Ontology-set intersection (paper definition, line 778 of sae_paper_v4.tex)
    both_ann = 0
    shared = 0
    for e in edges:
        src = ann_table.get((e["source_layer"], e["source_feature"]), frozenset())
        tgt = ann_table.get((e["target_layer"], e["target_feature"]), frozenset())
        if src and tgt:
            both_ann += 1
            if src & tgt:
                shared += 1
    shared_frac = (shared / both_ann) if both_ann else None

    # Single-label equality (coarser) and driver-gene overlap (orthogonal signal)
    label_match = sum(1 for e in edges
                      if e.get("source_label") and e.get("target_label")
                      and e["source_label"] == e["target_label"]) / len(edges)
    gene_overlap = 0
    both_genes = 0
    for e in edges:
        sg = set(e.get("source_genes") or [])
        tg = set(e.get("target_genes") or [])
        if sg and tg:
            both_genes += 1
            if sg & tg:
                gene_overlap += 1
    return dict(
        n_edges=len(edges),
        n_source_features=len(src_feats),
        n_target_features=len(tgt_feats),
        mean_abs_d=mean(abs_ds),
        median_abs_d=median(abs_ds),
        frac_abs_d_gt_1=sum(1 for x in abs_ds if x > 1.0) / len(abs_ds),
        frac_abs_d_gt_2=sum(1 for x in abs_ds if x > 2.0) / len(abs_ds),
        inhibitory_frac=inhib,
        shared_ontology_frac=shared_frac,
        shared_ontology_denom=both_ann,
        label_match_frac=label_match,
        gene_overlap_frac=(gene_overlap / both_genes) if both_genes else None,
        gene_overlap_denom=both_genes,
    )


def main():
    with open(EDGES_PATH) as f:
        all_edges = json.load(f)

    out = {}
    for cond, edges in all_edges.items():
        print(f"[{cond}] {len(edges)} raw edges; loading annotations...", flush=True)
        ann_table = build_feature_annotation_table(cond)
        per_threshold = {}
        for t in THRESHOLDS:
            sub = [e for e in edges if abs(e["cohens_d"]) >= t and e.get("consistency", 1) >= 0.7]
            per_threshold[f"{t}"] = metrics(sub, ann_table)
            m = per_threshold[f"{t}"]
            print(f"  |d|>={t}: n={m['n_edges']}, "
                  f"shared_onto={m.get('shared_ontology_frac') or 0:.3f}, "
                  f"inhib={m.get('inhibitory_frac', 0):.3f}, "
                  f"mean|d|={m.get('mean_abs_d', 0):.3f}", flush=True)
        out[cond] = per_threshold

    # Note the caveat: step1 edges were filtered at |d|>=0.5 during circuit
    # tracing. Thresholds below 0.5 here do NOT recover edges that were filtered
    # out upstream; they only show that the surviving edges are essentially all
    # above 0.5.
    out["_note"] = (
        "Edges in step1_annotated_edges.json were generated with an initial |d|>=0.5 "
        "and consistency>=0.7 filter during circuit tracing. Thresholds below 0.5 in "
        "this sweep therefore cannot be interpreted as adding lower-confidence edges; "
        "they only subset the existing graph. A complete low-threshold sweep requires "
        "re-running circuit_graph.json construction with per-edge raw deltas, which is "
        "deferred to the re-tracing pass (E5/E6). For the revision we report the "
        "stable-range behavior of metrics at |d|>={0.5, 0.7, 1.0}."
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
