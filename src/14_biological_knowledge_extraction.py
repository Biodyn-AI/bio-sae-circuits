#!/usr/bin/env python3
"""
Phase 5: Systematic Biological Knowledge Extraction

Systematically extract, validate, and categorize biological knowledge
from ~97K causal circuit edges across 4 conditions (K562/K562, K562/Multi,
TS/Multi Geneformer + scGPT).

Steps:
  1. Full Circuit Annotation — annotate all edges with source+target domains
  2. Cross-Model Consensus Graph — find domain pairs conserved across models
  3. Novel Relationship Discovery — find edges connecting unlinked domains
  4. Biological Process Hierarchy — reconstruct temporal ordering
  5. Cell-Type-Specific Circuit Activation — tissue enrichment analysis

Prerequisites:
  - Phase 4 circuit_graph.json from all 4 conditions (produced by 13_causal_circuit_tracing.py)
  - Feature annotations from bio-sae Phase 1 (feature_annotations.json per layer)
  - Biological reference databases: GO BP, KEGG, Reactome gene sets (JSON files)

Configuration:
  Set BASE below to your subproject root directory.
  Set BIO_DB to the directory containing go_bp_gene_sets.json, kegg_gene_sets.json, etc.
  Adjust CIRCUIT_FILES and ANNOTATION_DIRS to match your experiment layout.
"""

import json
import os
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# Line-buffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
OUT_DIR = os.path.join(BASE, "experiments/phase5_knowledge_extraction")
os.makedirs(OUT_DIR, exist_ok=True)

# Circuit graph files
CIRCUIT_FILES = {
    "K562_K562_GF": os.path.join(BASE, "experiments/phase1_k562/circuit_tracing/circuit_graph.json"),
    "K562_Multi_GF": os.path.join(BASE, "experiments/phase3_multitissue/circuit_tracing/circuit_graph.json"),
    "TS_Multi_GF": os.path.join(BASE, "experiments/phase3_multitissue/circuit_tracing_ts_cells/circuit_graph.json"),
    "scGPT_TS_Multi": os.path.join(BASE, "experiments/scgpt_atlas/circuit_tracing/circuit_graph.json"),
}

# SAE annotation directories per condition
ANNOTATION_DIRS = {
    "K562_K562_GF": os.path.join(BASE, "experiments/phase1_k562/sae_models"),
    "K562_Multi_GF": os.path.join(BASE, "experiments/phase3_multitissue/sae_models"),
    "TS_Multi_GF": os.path.join(BASE, "experiments/phase3_multitissue/sae_models"),
    "scGPT_TS_Multi": os.path.join(BASE, "experiments/scgpt_atlas/sae_models"),
}

# Biological databases
BIO_DB = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/results/biological_impact/reference_edge_sets"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_annotations_for_condition(annotation_base_dir):
    """Build lookup tables from feature_annotations.json + feature_catalog.json.

    Returns:
        label_lookup: {(layer, feature_idx) → best_GO_BP_term}
        gene_lookup:  {(layer, feature_idx) → [gene_name, ...]}
    """
    label_lookup = {}
    gene_lookup = {}

    for entry in sorted(os.listdir(annotation_base_dir)):
        if not entry.startswith("layer") or not os.path.isdir(
            os.path.join(annotation_base_dir, entry)
        ):
            continue

        # Parse layer number from "layer00_x4_k32"
        try:
            layer = int(entry.split("_")[0].replace("layer", ""))
        except ValueError:
            continue

        ann_path = os.path.join(annotation_base_dir, entry, "feature_annotations.json")
        cat_path = os.path.join(annotation_base_dir, entry, "feature_catalog.json")

        if not os.path.exists(ann_path):
            continue

        with open(ann_path) as f:
            ann_data = json.load(f)
        feature_annotations = ann_data.get("feature_annotations", {})

        # Load gene lists from catalog
        feat_genes = {}
        if os.path.exists(cat_path):
            with open(cat_path) as f:
                catalog = json.load(f)
            for feat in catalog.get("features", []):
                fi = feat["feature_idx"]
                top_genes = [g["gene_name"] for g in feat.get("top_genes", [])[:20]]
                feat_genes[fi] = top_genes

        for feat_idx_str, anns in feature_annotations.items():
            feat_idx = int(feat_idx_str)

            # Best GO_BP term (lowest p_adjusted)
            best_term = None
            best_p = 1.0
            for a in anns:
                if a.get("ontology") == "GO_BP" and a.get("p_adjusted", 1.0) < best_p:
                    best_p = a["p_adjusted"]
                    best_term = a["term"]

            if best_term:
                label_lookup[(layer, feat_idx)] = best_term

            if feat_idx in feat_genes:
                gene_lookup[(layer, feat_idx)] = feat_genes[feat_idx]

    return label_lookup, gene_lookup


# ════════════════════════════════════════════════════════════════════════════════
# STEP 1: Full Circuit Annotation
# ════════════════════════════════════════════════════════════════════════════════
def step1_full_annotation():
    """Annotate all ~97K edges with source and target biological domains."""

    out_path = os.path.join(OUT_DIR, "step1_annotated_edges.json")
    summary_path = os.path.join(OUT_DIR, "step1_domain_summary.json")

    if os.path.exists(out_path) and os.path.exists(summary_path):
        print("Step 1: Already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("=" * 60)
    print("STEP 1: Full Circuit Annotation")
    print("=" * 60)

    all_annotated = {}
    total_edges = 0
    total_annotated_both = 0

    for cond_name, graph_path in CIRCUIT_FILES.items():
        print(f"\n  Loading {cond_name}...")

        with open(graph_path) as f:
            graph = json.load(f)

        edges = graph["edges"]
        n_edges = len(edges)
        print(f"    {n_edges} edges")

        # Load annotations for this condition
        ann_dir = ANNOTATION_DIRS[cond_name]
        print(f"    Loading annotations from {ann_dir}...")
        label_lookup, gene_lookup = load_annotations_for_condition(ann_dir)
        print(f"    Label lookup: {len(label_lookup)} features annotated")
        print(f"    Gene lookup:  {len(gene_lookup)} features with gene lists")

        # Annotate each edge
        annotated_edges = []
        for edge in edges:
            source_label = edge.get("source_label", "unknown")
            src_key = (edge["source_layer"], edge["source_feature"])
            tgt_key = (edge["target_layer"], edge["target_feature"])
            target_label = label_lookup.get(tgt_key, None)

            annotated_edge = {
                "source_layer": edge["source_layer"],
                "source_feature": edge["source_feature"],
                "source_label": source_label,
                "target_layer": edge["target_layer"],
                "target_feature": edge["target_feature"],
                "target_label": target_label if target_label else "unannotated",
                "cohens_d": edge["cohens_d"],
                "consistency": edge["consistency"],
                "mean_delta": edge["mean_delta"],
                "source_genes": gene_lookup.get(src_key, []),
                "target_genes": gene_lookup.get(tgt_key, []),
            }
            annotated_edges.append(annotated_edge)

            if source_label != "unknown" and target_label:
                total_annotated_both += 1

        total_edges += n_edges
        all_annotated[cond_name] = annotated_edges

        n_target_ann = sum(
            1 for e in annotated_edges if e["target_label"] != "unannotated"
        )
        print(
            f"    Targets annotated: {n_target_ann}/{n_edges} ({100*n_target_ann/n_edges:.1f}%)"
        )

    print(
        f"\n  Total: {total_edges} edges, {total_annotated_both} with both domains annotated"
    )

    # ── Compute domain pair frequencies ──
    domain_pairs = defaultdict(
        lambda: {"count": 0, "conditions": set(), "abs_d_values": []}
    )
    domain_counts = defaultdict(
        lambda: {
            "as_source": 0,
            "as_target": 0,
            "conditions_source": set(),
            "conditions_target": set(),
        }
    )

    for cond_name, edges in all_annotated.items():
        for e in edges:
            sl = e["source_label"]
            tl = e["target_label"]
            if sl != "unknown" and tl != "unannotated":
                pair_key = f"{sl} -> {tl}"
                domain_pairs[pair_key]["count"] += 1
                domain_pairs[pair_key]["conditions"].add(cond_name)
                domain_pairs[pair_key]["abs_d_values"].append(abs(e["cohens_d"]))

                domain_counts[sl]["as_source"] += 1
                domain_counts[sl]["conditions_source"].add(cond_name)
                domain_counts[tl]["as_target"] += 1
                domain_counts[tl]["conditions_target"].add(cond_name)

    # Finalize
    domain_pairs_final = {}
    for pair_key, info in domain_pairs.items():
        domain_pairs_final[pair_key] = {
            "count": info["count"],
            "n_conditions": len(info["conditions"]),
            "conditions": sorted(info["conditions"]),
            "mean_abs_d": float(np.mean(info["abs_d_values"])),
        }
    domain_pairs_sorted = dict(
        sorted(domain_pairs_final.items(), key=lambda x: -x[1]["count"])
    )

    domain_counts_final = {}
    for domain, info in domain_counts.items():
        domain_counts_final[domain] = {
            "as_source": info["as_source"],
            "as_target": info["as_target"],
            "total": info["as_source"] + info["as_target"],
            "n_conditions_source": len(info["conditions_source"]),
            "n_conditions_target": len(info["conditions_target"]),
        }
    domain_counts_sorted = dict(
        sorted(domain_counts_final.items(), key=lambda x: -x[1]["total"])
    )

    # Print top domain pairs
    print("\n  Top 20 domain pairs:")
    for i, (pair, info) in enumerate(list(domain_pairs_sorted.items())[:20]):
        print(
            f"    {i+1}. {pair}: {info['count']} edges, "
            f"{info['n_conditions']} cond, mean|d|={info['mean_abs_d']:.2f}"
        )

    print("\n  Top 20 hub domains:")
    for i, (domain, info) in enumerate(list(domain_counts_sorted.items())[:20]):
        print(
            f"    {i+1}. {domain}: {info['total']} total "
            f"({info['as_source']} src, {info['as_target']} tgt)"
        )

    # ── Save ──
    print("\n  Saving Step 1 results...")
    with open(out_path, "w") as f:
        json.dump(all_annotated, f, indent=1, default=_json_default)

    summary = {
        "total_edges": total_edges,
        "total_annotated_both": total_annotated_both,
        "annotation_rate": total_annotated_both / total_edges if total_edges else 0,
        "n_unique_domain_pairs": len(domain_pairs_final),
        "n_unique_domains": len(domain_counts_final),
        "top_domain_pairs": dict(list(domain_pairs_sorted.items())[:50]),
        "top_hub_domains": dict(list(domain_counts_sorted.items())[:50]),
        "per_condition_stats": {
            cond: {
                "n_edges": len(edges),
                "n_both_annotated": sum(
                    1
                    for e in edges
                    if e["source_label"] != "unknown"
                    and e["target_label"] != "unannotated"
                ),
                "n_source_only": sum(
                    1
                    for e in edges
                    if e["source_label"] != "unknown"
                    and e["target_label"] == "unannotated"
                ),
                "n_target_only": sum(
                    1
                    for e in edges
                    if e["source_label"] == "unknown"
                    and e["target_label"] != "unannotated"
                ),
                "n_neither": sum(
                    1
                    for e in edges
                    if e["source_label"] == "unknown"
                    and e["target_label"] == "unannotated"
                ),
            }
            for cond, edges in all_annotated.items()
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"  Step 1 complete. Saved to {out_path}")
    return all_annotated


# ════════════════════════════════════════════════════════════════════════════════
# STEP 2: Cross-Model Consensus Graph
# ════════════════════════════════════════════════════════════════════════════════
def step2_consensus_graph(all_annotated):
    """Find domain pairs conserved across Geneformer AND scGPT."""

    out_path = os.path.join(OUT_DIR, "step2_consensus_graph.json")
    if os.path.exists(out_path):
        print("\nStep 2: Already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("\n" + "=" * 60)
    print("STEP 2: Cross-Model Consensus Graph")
    print("=" * 60)

    GF_CONDITIONS = ["K562_K562_GF", "K562_Multi_GF", "TS_Multi_GF"]
    SCGPT_CONDITIONS = ["scGPT_TS_Multi"]

    def get_domain_pairs(edges):
        """Extract unique (source, target) domain pairs with stats."""
        pairs = defaultdict(
            lambda: {"count": 0, "abs_d_values": [], "consistency_values": []}
        )
        for e in edges:
            if e["source_label"] != "unknown" and e["target_label"] != "unannotated":
                key = (e["source_label"], e["target_label"])
                pairs[key]["count"] += 1
                pairs[key]["abs_d_values"].append(abs(e["cohens_d"]))
                pairs[key]["consistency_values"].append(e["consistency"])
        return pairs

    # Get pairs for each condition
    cond_pairs = {}
    for cond in GF_CONDITIONS + SCGPT_CONDITIONS:
        cond_pairs[cond] = get_domain_pairs(all_annotated[cond])
        print(f"  {cond}: {len(cond_pairs[cond])} unique domain pairs")

    # Geneformer union
    gf_union = set()
    for cond in GF_CONDITIONS:
        gf_union.update(cond_pairs[cond].keys())
    print(f"  GF union: {len(gf_union)} unique pairs")

    # scGPT pairs
    scgpt_set = set(cond_pairs["scGPT_TS_Multi"].keys())
    print(f"  scGPT: {len(scgpt_set)} unique pairs")

    # Consensus = GF ∩ scGPT
    consensus = gf_union & scgpt_set
    print(f"  Consensus (GF ∩ scGPT): {len(consensus)} pairs")

    # Build consensus details
    consensus_details = []
    for src, tgt in sorted(consensus):
        scgpt_info = cond_pairs["scGPT_TS_Multi"][(src, tgt)]
        detail = {
            "source_domain": src,
            "target_domain": tgt,
            "gf_conditions": [],
            "scgpt_mean_abs_d": float(np.mean(scgpt_info["abs_d_values"])),
            "scgpt_count": scgpt_info["count"],
        }

        gf_d_values = []
        for cond in GF_CONDITIONS:
            if (src, tgt) in cond_pairs[cond]:
                detail["gf_conditions"].append(cond)
                gf_d_values.extend(cond_pairs[cond][(src, tgt)]["abs_d_values"])

        detail["gf_mean_abs_d"] = float(np.mean(gf_d_values))
        detail["combined_mean_abs_d"] = float(
            np.mean(gf_d_values + scgpt_info["abs_d_values"])
        )
        detail["n_gf_conditions"] = len(detail["gf_conditions"])
        consensus_details.append(detail)

    consensus_details.sort(key=lambda x: -x["combined_mean_abs_d"])

    # High-confidence: both models > 1.0
    high_confidence = [
        d
        for d in consensus_details
        if d["gf_mean_abs_d"] > 1.0 and d["scgpt_mean_abs_d"] > 1.0
    ]
    print(f"  High-confidence (both |d|>1.0): {len(high_confidence)} pairs")

    # ── Permutation test ──
    print("  Running permutation test (1000 iterations)...")
    all_domains_scgpt = list(set(d for pair in scgpt_set for d in pair))

    rng = np.random.RandomState(42)
    perm_counts = []
    for _ in range(1000):
        shuffled = rng.permutation(all_domains_scgpt)
        domain_map = dict(zip(all_domains_scgpt, shuffled))
        shuffled_scgpt = set(
            (domain_map.get(s, s), domain_map.get(t, t)) for (s, t) in scgpt_set
        )
        perm_counts.append(len(gf_union & shuffled_scgpt))

    perm_counts = np.array(perm_counts)
    p_value = float(np.mean(perm_counts >= len(consensus)))
    expected = float(np.mean(perm_counts))
    enrichment = len(consensus) / max(expected, 1)

    print(
        f"  Permutation: expected={expected:.1f}, observed={len(consensus)}, "
        f"enrichment={enrichment:.1f}x, p={p_value:.4f}"
    )

    # Print top consensus pairs
    print(f"\n  Top 20 consensus pairs:")
    for i, d in enumerate(consensus_details[:20]):
        print(
            f"    {i+1}. {d['source_domain']} -> {d['target_domain']}: "
            f"GF|d|={d['gf_mean_abs_d']:.2f}, scGPT|d|={d['scgpt_mean_abs_d']:.2f}, "
            f"in {d['n_gf_conditions']} GF conditions"
        )

    # Save
    result = {
        "n_consensus": len(consensus),
        "n_high_confidence": len(high_confidence),
        "n_gf_union": len(gf_union),
        "n_scgpt": len(scgpt_set),
        "permutation_p_value": p_value,
        "permutation_expected": expected,
        "enrichment_ratio": enrichment,
        "consensus_pairs": consensus_details,
        "high_confidence_pairs": high_confidence,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    print(f"  Step 2 complete. Saved to {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# STEP 3: Novel Relationship Discovery
# ════════════════════════════════════════════════════════════════════════════════
def step3_novel_discovery(all_annotated):
    """Find strong edges connecting domains NOT linked in existing databases."""

    out_path = os.path.join(OUT_DIR, "step3_novel_candidates.json")
    if os.path.exists(out_path):
        print("\nStep 3: Already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("\n" + "=" * 60)
    print("STEP 3: Novel Relationship Discovery")
    print("=" * 60)

    # Load biological reference databases
    print("  Loading biological databases...")

    with open(os.path.join(BIO_DB, "go_bp_gene_sets.json")) as f:
        go_bp = json.load(f)
    with open(os.path.join(BIO_DB, "kegg_gene_sets.json")) as f:
        kegg = json.load(f)
    with open(os.path.join(BIO_DB, "reactome_gene_sets.json")) as f:
        reactome = json.load(f)

    print(f"    GO_BP: {len(go_bp)} terms, KEGG: {len(kegg)}, Reactome: {len(reactome)}")

    # Combine all gene sets
    all_gene_sets = {}
    all_gene_sets.update(go_bp)
    all_gene_sets.update(kegg)
    all_gene_sets.update(reactome)

    # Collect all unique domains from circuit edges
    all_domains = set()
    for cond, edges in all_annotated.items():
        for e in edges:
            if e["source_label"] != "unknown":
                all_domains.add(e["source_label"])
            if e["target_label"] != "unannotated":
                all_domains.add(e["target_label"])
    print(f"    {len(all_domains)} unique domains in circuits")

    # Match circuit domains to gene set terms
    gene_set_keys = set(all_gene_sets.keys())
    domain_to_genes = {}
    for domain in all_domains:
        if domain in gene_set_keys:
            domain_to_genes[domain] = set(all_gene_sets[domain])
        else:
            # Try matching by extracting GO/KEGG/Reactome ID from the domain name
            # Domain format: "DNA Repair (GO:0006281)" or just a term name
            for term_key, genes in all_gene_sets.items():
                # Check if one contains the other
                if domain in term_key or term_key in domain:
                    domain_to_genes[domain] = set(genes)
                    break

    print(
        f"    Matched {len(domain_to_genes)}/{len(all_domains)} domains to gene sets"
    )

    # Build known-link set: two domains are "linked" if they share >=3 genes
    print("  Building known-biology reference graph (>=3 shared genes)...")
    domains_with_genes = list(domain_to_genes.keys())
    n_dom = len(domains_with_genes)
    print(f"    Checking {n_dom*(n_dom-1)//2} domain pairs...")

    known_links = set()
    for i in range(n_dom):
        for j in range(i + 1, n_dom):
            d1, d2 = domains_with_genes[i], domains_with_genes[j]
            shared = domain_to_genes[d1] & domain_to_genes[d2]
            if len(shared) >= 3:
                known_links.add((d1, d2))
                known_links.add((d2, d1))

    print(f"    {len(known_links)//2} known domain-domain links (>=3 shared genes)")

    # Find novel candidates: edges between unlinked domains
    print("  Scanning for novel relationships...")
    novel_candidates = []

    for cond, edges in all_annotated.items():
        for e in edges:
            sl = e["source_label"]
            tl = e["target_label"]
            if sl == "unknown" or tl == "unannotated" or sl == tl:
                continue

            is_known = (sl, tl) in known_links

            if not is_known:
                novel_candidates.append(
                    {
                        "source_domain": sl,
                        "target_domain": tl,
                        "condition": cond,
                        "cohens_d": e["cohens_d"],
                        "abs_d": abs(e["cohens_d"]),
                        "consistency": e["consistency"],
                        "source_layer": e["source_layer"],
                        "target_layer": e["target_layer"],
                        "source_genes": e.get("source_genes", []),
                        "target_genes": e.get("target_genes", []),
                    }
                )

    print(f"    {len(novel_candidates)} novel candidate edges")

    # Aggregate by domain pair
    novel_pairs = defaultdict(
        lambda: {
            "edges": [],
            "conditions": set(),
            "max_abs_d": 0,
            "source_genes_union": set(),
            "target_genes_union": set(),
        }
    )

    for nc in novel_candidates:
        key = (nc["source_domain"], nc["target_domain"])
        novel_pairs[key]["edges"].append(nc)
        novel_pairs[key]["conditions"].add(nc["condition"])
        novel_pairs[key]["max_abs_d"] = max(
            novel_pairs[key]["max_abs_d"], nc["abs_d"]
        )
        novel_pairs[key]["source_genes_union"].update(nc.get("source_genes", []))
        novel_pairs[key]["target_genes_union"].update(nc.get("target_genes", []))

    # Build ranked list
    novel_pairs_list = []
    for (src, tgt), info in novel_pairs.items():
        abs_d_values = [e["abs_d"] for e in info["edges"]]
        shared_genes = info["source_genes_union"] & info["target_genes_union"]

        novel_pairs_list.append(
            {
                "source_domain": src,
                "target_domain": tgt,
                "n_edges": len(info["edges"]),
                "n_conditions": len(info["conditions"]),
                "conditions": sorted(info["conditions"]),
                "max_abs_d": info["max_abs_d"],
                "mean_abs_d": float(np.mean(abs_d_values)),
                "shared_genes": sorted(shared_genes),
                "n_shared_genes": len(shared_genes),
                "source_genes": sorted(info["source_genes_union"]),
                "target_genes": sorted(info["target_genes_union"]),
            }
        )

    # Rank by cross-model presence, then max |d|
    novel_pairs_list.sort(key=lambda x: (-x["n_conditions"], -x["max_abs_d"]))

    # Print top 20
    print(f"\n  Top 20 novel domain pairs:")
    for i, np_ in enumerate(novel_pairs_list[:20]):
        print(
            f"    {i+1}. {np_['source_domain']} -> {np_['target_domain']}: "
            f"|d|={np_['max_abs_d']:.2f}, {np_['n_edges']} edges, "
            f"{np_['n_conditions']} cond, {np_['n_shared_genes']} shared genes"
        )

    # Save
    result = {
        "n_novel_edges": len(novel_candidates),
        "n_novel_pairs": len(novel_pairs_list),
        "n_known_links": len(known_links) // 2,
        "n_domains_matched": len(domain_to_genes),
        "top_novel_pairs": novel_pairs_list[:100],
        "all_novel_pairs": novel_pairs_list,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    print(f"  Step 3 complete. Saved to {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# STEP 4: Biological Process Hierarchy
# ════════════════════════════════════════════════════════════════════════════════
def step4_hierarchy(all_annotated):
    """Reconstruct temporal ordering of biological processes from directed graph."""

    out_path = os.path.join(OUT_DIR, "step4_hierarchy.json")
    if os.path.exists(out_path):
        print("\nStep 4: Already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("\n" + "=" * 60)
    print("STEP 4: Biological Process Hierarchy")
    print("=" * 60)

    import networkx as nx

    # Build meta-graph: nodes = domains, edges = aggregated stats
    meta_edges = defaultdict(
        lambda: {"d_values": [], "layer_deltas": [], "conditions": set()}
    )

    for cond, edges in all_annotated.items():
        for e in edges:
            sl = e["source_label"]
            tl = e["target_label"]
            if sl == "unknown" or tl == "unannotated" or sl == tl:
                continue
            key = (sl, tl)
            meta_edges[key]["d_values"].append(e["cohens_d"])
            meta_edges[key]["layer_deltas"].append(
                e["target_layer"] - e["source_layer"]
            )
            meta_edges[key]["conditions"].add(cond)

    # Build networkx DiGraph
    G = nx.DiGraph()
    for (src, tgt), info in meta_edges.items():
        G.add_edge(
            src,
            tgt,
            mean_d=float(np.mean(info["d_values"])),
            mean_abs_d=float(np.mean(np.abs(info["d_values"]))),
            n_edges=len(info["d_values"]),
            mean_layer_delta=float(np.mean(info["layer_deltas"])),
            n_conditions=len(info["conditions"]),
        )

    print(f"  Meta-graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── PageRank ──
    print("  Computing PageRank...")
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except Exception:
        pagerank = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

    pr_sorted = sorted(pagerank.items(), key=lambda x: -x[1])
    print("  Top 10 PageRank domains:")
    for i, (domain, pr) in enumerate(pr_sorted[:10]):
        print(
            f"    {i+1}. {domain}: PR={pr:.4f}, "
            f"in={G.in_degree(domain)}, out={G.out_degree(domain)}"
        )

    # ── Feedback loops (A->B and B->A) ──
    print("\n  Identifying feedback loops...")
    feedback_loops = []
    for u, v in G.edges():
        if G.has_edge(v, u) and u < v:  # avoid duplicates
            uv_data = G.edges[u, v]
            vu_data = G.edges[v, u]
            feedback_loops.append(
                {
                    "domain_a": u,
                    "domain_b": v,
                    "a_to_b_mean_d": uv_data["mean_abs_d"],
                    "b_to_a_mean_d": vu_data["mean_abs_d"],
                    "a_to_b_layer_delta": uv_data["mean_layer_delta"],
                    "b_to_a_layer_delta": vu_data["mean_layer_delta"],
                }
            )

    feedback_loops.sort(key=lambda x: -(x["a_to_b_mean_d"] + x["b_to_a_mean_d"]))
    print(f"    {len(feedback_loops)} feedback loops found")
    for i, fl in enumerate(feedback_loops[:10]):
        print(
            f"    {i+1}. {fl['domain_a']} <-> {fl['domain_b']}: "
            f"|d|={fl['a_to_b_mean_d']:.2f}/{fl['b_to_a_mean_d']:.2f}"
        )

    # ── Layer centrality ──
    print("\n  Computing layer centrality...")
    layer_domains = defaultdict(lambda: defaultdict(int))

    for cond, edges in all_annotated.items():
        for e in edges:
            if e["source_label"] != "unknown":
                layer_domains[e["source_layer"]][e["source_label"]] += 1
            if e["target_label"] != "unannotated":
                layer_domains[e["target_layer"]][e["target_label"]] += 1

    # Mean layer per domain
    domain_layers = defaultdict(list)
    for layer, domains in layer_domains.items():
        for domain, count in domains.items():
            domain_layers[domain].extend([layer] * count)

    domain_mean_layer = {}
    for domain, layers in domain_layers.items():
        domain_mean_layer[domain] = {
            "mean_layer": float(np.mean(layers)),
            "std_layer": float(np.std(layers)),
            "min_layer": int(np.min(layers)),
            "max_layer": int(np.max(layers)),
            "n_occurrences": len(layers),
        }

    early_domains = sorted(domain_mean_layer.items(), key=lambda x: x[1]["mean_layer"])
    late_domains = sorted(
        domain_mean_layer.items(), key=lambda x: -x[1]["mean_layer"]
    )

    print("  Earliest domains (lowest mean layer):")
    for i, (d, info) in enumerate(early_domains[:10]):
        print(
            f"    {i+1}. {d}: mean_L={info['mean_layer']:.1f} "
            f"(L{info['min_layer']}-L{info['max_layer']})"
        )

    print("  Latest domains (highest mean layer):")
    for i, (d, info) in enumerate(late_domains[:10]):
        print(
            f"    {i+1}. {d}: mean_L={info['mean_layer']:.1f} "
            f"(L{info['min_layer']}-L{info['max_layer']})"
        )

    # ── Topological sort on strong subgraph ──
    print("\n  Attempting topological sort on strong subgraph (|d|>1.0, n>=3)...")
    strong_G = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data["mean_abs_d"] > 1.0 and data["n_edges"] >= 3:
            strong_G.add_edge(u, v, **data)

    print(
        f"    Strong subgraph: {strong_G.number_of_nodes()} nodes, "
        f"{strong_G.number_of_edges()} edges"
    )

    is_dag = nx.is_directed_acyclic_graph(strong_G)
    topo_order = None
    sccs_info = []

    if is_dag:
        topo_order = list(nx.topological_sort(strong_G))
        print(f"    DAG! Topological order ({len(topo_order)} nodes)")
    else:
        sccs = [
            c for c in nx.strongly_connected_components(strong_G) if len(c) > 1
        ]
        print(
            f"    Not a DAG. {len(sccs)} strongly connected components with cycles"
        )
        for i, scc in enumerate(sorted(sccs, key=len, reverse=True)[:5]):
            scc_list = sorted(scc)
            print(f"      SCC {i+1} ({len(scc)} nodes): {scc_list[:3]}...")
            sccs_info.append({"size": len(scc), "members": scc_list[:10]})

        # Condensation
        condensed = nx.condensation(strong_G)
        topo_condensed = list(nx.topological_sort(condensed))
        print(f"    Condensed DAG: {condensed.number_of_nodes()} meta-nodes")

    # ── Validate: DDR -> cell cycle ordering ──
    print("\n  Validation: DDR -> checkpoint -> arrest ordering...")
    ddr_keywords = ["DNA", "repair", "DDR", "damage"]
    cc_keywords = ["cell cycle", "checkpoint", "mitotic", "mitosis", "division"]

    ddr_domains = [
        d
        for d in G.nodes()
        if any(kw.lower() in d.lower() for kw in ddr_keywords)
    ]
    cc_domains = [
        d
        for d in G.nodes()
        if any(kw.lower() in d.lower() for kw in cc_keywords)
    ]

    print(f"    DDR domains ({len(ddr_domains)}): {ddr_domains[:5]}")
    print(f"    Cell cycle domains ({len(cc_domains)}): {cc_domains[:5]}")

    ddr_to_cc = []
    for ddr in ddr_domains:
        for cc in cc_domains:
            if G.has_edge(ddr, cc):
                data = G.edges[ddr, cc]
                ddr_to_cc.append(
                    {
                        "from": ddr,
                        "to": cc,
                        "mean_abs_d": data["mean_abs_d"],
                        "mean_layer_delta": data["mean_layer_delta"],
                    }
                )

    if ddr_to_cc:
        print(f"    Found {len(ddr_to_cc)} DDR -> cell cycle edges:")
        for e in ddr_to_cc[:10]:
            print(
                f"      {e['from']} -> {e['to']}: "
                f"|d|={e['mean_abs_d']:.2f}, delta_L={e['mean_layer_delta']:.1f}"
            )
    else:
        print("    No direct DDR -> cell cycle edges found")

    # Save
    result = {
        "n_meta_nodes": G.number_of_nodes(),
        "n_meta_edges": G.number_of_edges(),
        "pagerank_top50": [
            {
                "domain": d,
                "pagerank": float(pr),
                "in_degree": G.in_degree(d),
                "out_degree": G.out_degree(d),
            }
            for d, pr in pr_sorted[:50]
        ],
        "feedback_loops": feedback_loops[:50],
        "n_feedback_loops": len(feedback_loops),
        "early_domains": [{"domain": d, **info} for d, info in early_domains[:30]],
        "late_domains": [{"domain": d, **info} for d, info in late_domains[:30]],
        "domain_mean_layers": domain_mean_layer,
        "strong_subgraph": {
            "n_nodes": strong_G.number_of_nodes(),
            "n_edges": strong_G.number_of_edges(),
            "is_dag": is_dag,
            "sccs": sccs_info,
        },
        "ddr_to_cell_cycle": ddr_to_cc,
        "topological_order": topo_order[:30] if topo_order else None,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    print(f"  Step 4 complete. Saved to {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# STEP 5: Cell-Type-Specific Circuit Activation
# ════════════════════════════════════════════════════════════════════════════════
def step5_celltype_circuits(all_annotated):
    """Determine which circuits activate in which cell types via tissue-keyword
    classification and enrichment analysis."""

    out_path = os.path.join(OUT_DIR, "step5_celltype_circuits.json")
    if os.path.exists(out_path):
        print("\nStep 5: Already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("\n" + "=" * 60)
    print("STEP 5: Cell-Type-Specific Circuit Activation")
    print("=" * 60)

    from scipy.stats import fisher_exact

    # Tissue keyword classifiers
    tissue_keywords = {
        "immune": [
            "immune",
            "T cell",
            "B cell",
            "lymphocyte",
            "leukocyte",
            "cytokine",
            "interferon",
            "interleukin",
            "inflammatory",
            "innate immune",
            "adaptive immune",
            "antigen",
            "MHC",
            "complement",
            "natural killer",
            "macrophage",
            "dendritic",
        ],
        "kidney": [
            "kidney",
            "renal",
            "nephron",
            "glomerular",
            "tubular",
            "urine",
            "filtration",
            "ion transport",
            "water transport",
        ],
        "lung": [
            "lung",
            "respiratory",
            "surfactant",
            "alveolar",
            "bronchial",
            "pulmonary",
            "gas exchange",
            "airway",
        ],
        "blood": [
            "erythrocyte",
            "hemoglobin",
            "hematopoietic",
            "platelet",
            "coagulation",
            "blood",
            "myeloid",
        ],
        "universal": [
            "ribosom",
            "mitochond",
            "translation",
            "transcription",
            "cell cycle",
            "apoptosis",
            "DNA repair",
            "RNA processing",
            "protein folding",
            "ubiquitin",
            "proteasom",
            "metaboli",
        ],
    }

    def classify_domain(domain_name):
        domain_lower = domain_name.lower()
        matches = []
        for tissue, keywords in tissue_keywords.items():
            for kw in keywords:
                if kw.lower() in domain_lower:
                    matches.append(tissue)
                    break
        return matches if matches else ["unclassified"]

    # Classify all domains present in multi-tissue conditions
    all_domains = set()
    for cond in ["TS_Multi_GF", "scGPT_TS_Multi", "K562_K562_GF"]:
        for e in all_annotated[cond]:
            if e["source_label"] != "unknown":
                all_domains.add(e["source_label"])
            if e["target_label"] != "unannotated":
                all_domains.add(e["target_label"])

    domain_classifications = {}
    for domain in all_domains:
        domain_classifications[domain] = classify_domain(domain)

    tissue_counts = defaultdict(int)
    for domain, tissues in domain_classifications.items():
        for t in tissues:
            tissue_counts[t] += 1

    print("  Domain tissue classification:")
    for tissue, count in sorted(tissue_counts.items(), key=lambda x: -x[1]):
        print(f"    {tissue}: {count} domains")

    # ── Compare K562-only vs multi-tissue pairs ──
    k562_pairs = set()
    for e in all_annotated["K562_K562_GF"]:
        if e["source_label"] != "unknown" and e["target_label"] != "unannotated":
            k562_pairs.add((e["source_label"], e["target_label"]))

    ts_pairs = set()
    for cond in ["TS_Multi_GF", "scGPT_TS_Multi"]:
        for e in all_annotated[cond]:
            if e["source_label"] != "unknown" and e["target_label"] != "unannotated":
                ts_pairs.add((e["source_label"], e["target_label"]))

    ts_only = ts_pairs - k562_pairs
    shared_pairs = ts_pairs & k562_pairs
    k562_only = k562_pairs - ts_pairs

    print(f"\n  TS-only pairs: {len(ts_only)}")
    print(f"  Shared pairs:  {len(shared_pairs)}")
    print(f"  K562-only pairs: {len(k562_only)}")

    # ── Fisher's exact: tissue enrichment in TS-only vs shared circuits ──
    print("\n  Tissue enrichment (TS-only vs shared):")
    enrichment_results = {}
    for tissue in tissue_keywords.keys():
        ts_only_tissue = sum(
            1
            for (s, t) in ts_only
            if tissue in domain_classifications.get(s, [])
            or tissue in domain_classifications.get(t, [])
        )
        ts_only_other = len(ts_only) - ts_only_tissue

        shared_tissue = sum(
            1
            for (s, t) in shared_pairs
            if tissue in domain_classifications.get(s, [])
            or tissue in domain_classifications.get(t, [])
        )
        shared_other = len(shared_pairs) - shared_tissue

        table = [[ts_only_tissue, ts_only_other], [shared_tissue, shared_other]]

        if ts_only_tissue > 0 or shared_tissue > 0:
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            enrichment_results[tissue] = {
                "ts_only_tissue": ts_only_tissue,
                "ts_only_other": ts_only_other,
                "shared_tissue": shared_tissue,
                "shared_other": shared_other,
                "odds_ratio": float(odds_ratio),
                "p_value": float(p_value),
            }
            sig = " *" if p_value < 0.05 else ""
            print(
                f"    {tissue}: TS-only={ts_only_tissue}/{len(ts_only)}, "
                f"shared={shared_tissue}/{len(shared_pairs)}, "
                f"OR={odds_ratio:.2f}, p={p_value:.4f}{sig}"
            )

    # ── Classify circuits as universal vs tissue-specific ──
    universal_circuits = []
    tissue_specific_circuits = defaultdict(list)

    for src, tgt in ts_pairs:
        src_class = domain_classifications.get(src, ["unclassified"])
        tgt_class = domain_classifications.get(tgt, ["unclassified"])

        is_universal = "universal" in src_class or "universal" in tgt_class
        tissue_labels = set(src_class + tgt_class) - {"unclassified", "universal"}

        in_k562 = (src, tgt) in k562_pairs

        entry = {
            "source": src,
            "target": tgt,
            "in_k562": in_k562,
            "tissue_labels": sorted(tissue_labels),
        }

        if is_universal and not tissue_labels:
            universal_circuits.append(entry)
        elif tissue_labels:
            for t in tissue_labels:
                tissue_specific_circuits[t].append(entry)

    print(f"\n  Universal circuits: {len(universal_circuits)}")
    for tissue, circuits in sorted(tissue_specific_circuits.items()):
        k562_overlap = sum(1 for c in circuits if c["in_k562"])
        print(
            f"  {tissue}-specific: {len(circuits)} "
            f"({k562_overlap} also in K562)"
        )

    # ── Per-tissue circuit profiles (using edge counts across conditions) ──
    circuit_tissue_profile = defaultdict(
        lambda: {"n_edges": 0, "conditions": set(), "domain_pairs": set()}
    )

    for cond in ["TS_Multi_GF", "scGPT_TS_Multi"]:
        for e in all_annotated[cond]:
            sl = e["source_label"]
            tl = e["target_label"]
            if sl == "unknown" or tl == "unannotated":
                continue

            src_tissues = domain_classifications.get(sl, ["unclassified"])
            tgt_tissues = domain_classifications.get(tl, ["unclassified"])

            for tissue in set(src_tissues + tgt_tissues):
                circuit_tissue_profile[tissue]["n_edges"] += 1
                circuit_tissue_profile[tissue]["conditions"].add(cond)
                circuit_tissue_profile[tissue]["domain_pairs"].add((sl, tl))

    # Save
    result = {
        "n_ts_only_pairs": len(ts_only),
        "n_shared_pairs": len(shared_pairs),
        "n_k562_only_pairs": len(k562_only),
        "domain_classifications": {d: c for d, c in domain_classifications.items()},
        "tissue_counts": dict(tissue_counts),
        "enrichment_tests": enrichment_results,
        "n_universal_circuits": len(universal_circuits),
        "tissue_specific_counts": {
            t: len(c) for t, c in tissue_specific_circuits.items()
        },
        "universal_circuits_sample": universal_circuits[:50],
        "tissue_specific_sample": {
            t: circuits[:30] for t, circuits in tissue_specific_circuits.items()
        },
        "circuit_tissue_profile": {
            t: {
                "n_edges": info["n_edges"],
                "n_conditions": len(info["conditions"]),
                "n_domain_pairs": len(info["domain_pairs"]),
            }
            for t, info in circuit_tissue_profile.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)

    print(f"  Step 5 complete. Saved to {out_path}")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    print("Phase 5: Systematic Biological Knowledge Extraction")
    print(f"Output: {OUT_DIR}")
    print()

    # Step 1
    all_annotated = step1_full_annotation()

    # Step 2
    consensus = step2_consensus_graph(all_annotated)

    # Step 3
    novel = step3_novel_discovery(all_annotated)

    # Step 4
    hierarchy = step4_hierarchy(all_annotated)

    # Step 5
    celltype = step5_celltype_circuits(all_annotated)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Phase 5 complete in {elapsed/60:.1f} minutes")
    print(f"Output: {OUT_DIR}")
    print(f"{'=' * 60}")
