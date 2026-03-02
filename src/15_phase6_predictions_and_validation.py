#!/usr/bin/env python3
"""
Phase 6: Gene-Level Predictions, Perturbation Validation & Disease Mapping

Drills down from Phase 5 domain-level analysis to gene-level predictions,
validates against CRISPRi perturbation data, and maps to disease biology.

Steps:
  1. Gene-Level Prediction Extraction — extract gene→gene predictions from circuit edges
  2. Perturbation Response Validation — test predictions against Replogle CRISPRi data
  3. Disease Gene Mapping — map disease gene sets onto circuit hubs and features

Prerequisites:
  - Phase 5 outputs: step1_annotated_edges.json, step2_consensus_graph.json,
    step3_novel_candidates.json, step4_hierarchy.json (from 14_biological_knowledge_extraction.py)
  - Replogle CRISPRi K562 data (replogle_concat.h5ad, ~4 GB)
  - Biological reference databases: STRING PPI, TRRUST TF-target, GO BP gene sets

Configuration:
  Set BASE below to your subproject root directory.
  Set BIO_DB to the directory containing string_ppi_edges.json, go_bp_gene_sets.json.
  Set REPLOGLE_PATH and TRRUST_PATH to your local copies of these datasets.

Usage:
    python src/15_phase6_predictions_and_validation.py
"""

import json
import os
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from scipy import stats

# Line-buffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/subproject_42_sparse_autoencoder_biological_map"
P5_DIR = os.path.join(BASE, "experiments/phase5_knowledge_extraction")
OUT_DIR = os.path.join(BASE, "experiments/phase6_predictions_validation")
os.makedirs(OUT_DIR, exist_ok=True)

BIO_DB = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/results/biological_impact/reference_edge_sets"
REPLOGLE_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-nmi-paper/src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad"
TRRUST_PATH = "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/external/networks/trrust_human.tsv"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, default=_json_default, indent=1)
    print(f"  Saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


# ════════════════════════════════════════════════════════════════════════════════
# Step 1: Gene-Level Prediction Extraction
# ════════════════════════════════════════════════════════════════════════════════
def step1_gene_predictions():
    out_path = os.path.join(OUT_DIR, "step1_gene_predictions.json")
    if os.path.exists(out_path):
        print("Step 1 already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("=" * 70)
    print("STEP 1: Gene-Level Prediction Extraction")
    print("=" * 70)
    t0 = time.time()

    # Load Phase 5 data
    print("Loading Phase 5 annotated edges...")
    with open(os.path.join(P5_DIR, "step1_annotated_edges.json")) as f:
        all_edges = json.load(f)

    print("Loading consensus graph...")
    with open(os.path.join(P5_DIR, "step2_consensus_graph.json")) as f:
        consensus = json.load(f)

    print("Loading novel candidates...")
    with open(os.path.join(P5_DIR, "step3_novel_candidates.json")) as f:
        novel = json.load(f)

    # Build consensus and novel lookup sets
    consensus_pairs = set()
    for p in consensus['consensus_pairs']:
        consensus_pairs.add((p['source_domain'], p['target_domain']))

    novel_pairs = set()
    for p in novel['all_novel_pairs']:
        novel_pairs.add((p['source_domain'], p['target_domain']))

    # Load reference databases for validation
    print("Loading STRING PPI...")
    with open(os.path.join(BIO_DB, "string_ppi_edges.json")) as f:
        string_data = json.load(f)
    string_ppi = set()
    for pair in string_data['pairs_700']:
        string_ppi.add((pair[0], pair[1]))
        string_ppi.add((pair[1], pair[0]))
    print(f"  {len(string_ppi) // 2} STRING PPI edges (score >= 700)")

    print("Loading TRRUST...")
    trrust_edges = set()
    with open(TRRUST_PATH) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                trrust_edges.add((parts[0], parts[1]))
    print(f"  {len(trrust_edges)} TRRUST TF→target edges")

    print("Loading GO BP gene sets for co-annotation check...")
    with open(os.path.join(BIO_DB, "go_bp_gene_sets.json")) as f:
        go_bp = json.load(f)
    # Build gene → GO terms mapping
    gene_to_go = defaultdict(set)
    for term, genes in go_bp.items():
        for g in genes:
            gene_to_go[g].add(term)

    # Extract gene-pair predictions from circuit edges
    print("Extracting gene-pair predictions from all circuit edges...")
    gene_pair_evidence = defaultdict(lambda: {
        'edges': 0, 'conditions': set(), 'sum_abs_d': 0.0,
        'max_abs_d': 0.0, 'is_consensus': False, 'is_novel': False,
        'source_domains': set(), 'target_domains': set(),
        'sum_d': 0.0,  # track sign
    })

    total_edges = 0
    edges_with_genes = 0

    for cond, edges in all_edges.items():
        for edge in edges:
            total_edges += 1
            src_genes = edge.get('source_genes', [])
            tgt_genes = edge.get('target_genes', [])
            if not src_genes or not tgt_genes:
                continue
            edges_with_genes += 1

            d = edge['cohens_d']
            abs_d = abs(d)
            src_domain = edge.get('source_label', 'unknown')
            tgt_domain = edge.get('target_label', 'unannotated')
            domain_pair = (src_domain, tgt_domain)
            is_cons = domain_pair in consensus_pairs
            is_nov = domain_pair in novel_pairs

            # Use top-N genes weighted by rank (top genes are more important)
            # Weight top genes more: use top 10 from source, top 10 from target
            n_src = min(10, len(src_genes))
            n_tgt = min(10, len(tgt_genes))

            for i, sg in enumerate(src_genes[:n_src]):
                for j, tg in enumerate(tgt_genes[:n_tgt]):
                    if sg == tg:
                        continue
                    pair_key = (sg, tg)
                    ev = gene_pair_evidence[pair_key]
                    rank_weight = 1.0 / ((i + 1) * (j + 1))
                    ev['edges'] += 1
                    ev['conditions'].add(cond)
                    ev['sum_abs_d'] += abs_d * rank_weight
                    ev['sum_d'] += d * rank_weight
                    ev['max_abs_d'] = max(ev['max_abs_d'], abs_d)
                    if is_cons:
                        ev['is_consensus'] = True
                    if is_nov:
                        ev['is_novel'] = True
                    ev['source_domains'].add(src_domain)
                    ev['target_domains'].add(tgt_domain)

    print(f"  {total_edges} total edges, {edges_with_genes} with gene lists")
    print(f"  {len(gene_pair_evidence)} raw gene pairs extracted")

    # Filter: require ≥2 independent edges OR 1 edge with |d|>2
    filtered = {}
    for pair, ev in gene_pair_evidence.items():
        if ev['edges'] >= 2 or ev['max_abs_d'] > 2.0:
            filtered[pair] = ev

    print(f"  {len(filtered)} gene pairs after filtering (≥2 edges or |d|>2)")

    # Validate each pair against known biology
    print("Validating against STRING, TRRUST, GO co-annotation...")
    predictions = []
    n_string = 0
    n_trrust = 0
    n_go_shared = 0
    n_novel_pred = 0

    for (sg, tg), ev in filtered.items():
        in_string = (sg, tg) in string_ppi
        in_trrust = (sg, tg) in trrust_edges
        # GO co-annotation: share ≥2 GO BP terms
        shared_go = gene_to_go.get(sg, set()) & gene_to_go.get(tg, set())
        n_shared_go = len(shared_go)

        if in_string or in_trrust:
            status = 'confirmed'
            n_string += int(in_string)
            n_trrust += int(in_trrust)
        elif n_shared_go >= 2:
            status = 'plausible'
            n_go_shared += 1
        else:
            status = 'novel'
            n_novel_pred += 1

        mean_abs_d = ev['sum_abs_d'] / ev['edges'] if ev['edges'] > 0 else 0
        mean_d = ev['sum_d'] / ev['edges'] if ev['edges'] > 0 else 0

        predictions.append({
            'source_gene': sg,
            'target_gene': tg,
            'n_edges': ev['edges'],
            'n_conditions': len(ev['conditions']),
            'conditions': sorted(ev['conditions']),
            'mean_abs_d': round(mean_abs_d, 4),
            'mean_d': round(mean_d, 4),
            'max_abs_d': round(ev['max_abs_d'], 4),
            'is_consensus': ev['is_consensus'],
            'is_novel': ev['is_novel'],
            'validation_status': status,
            'in_string_ppi': in_string,
            'in_trrust': in_trrust,
            'n_shared_go_terms': n_shared_go,
        })

    # Sort by mean_abs_d descending
    predictions.sort(key=lambda x: x['mean_abs_d'], reverse=True)

    # Summary stats
    print(f"\n  Gene-pair prediction summary:")
    print(f"    Total predictions: {len(predictions)}")
    print(f"    Confirmed (STRING/TRRUST): {n_string + n_trrust - len([p for p in predictions if p['in_string_ppi'] and p['in_trrust']])}")
    print(f"      - In STRING PPI: {n_string}")
    print(f"      - In TRRUST: {n_trrust}")
    print(f"    Plausible (shared GO): {n_go_shared}")
    print(f"    Novel (no known link): {n_novel_pred}")
    print(f"    Consensus pairs: {sum(1 for p in predictions if p['is_consensus'])}")
    print(f"    Novel domain pairs: {sum(1 for p in predictions if p['is_novel'])}")

    # Top 20 predictions
    print(f"\n  Top 20 predictions by weighted |d|:")
    for i, p in enumerate(predictions[:20]):
        print(f"    {i+1}. {p['source_gene']} → {p['target_gene']}: "
              f"|d|={p['mean_abs_d']:.2f}, edges={p['n_edges']}, "
              f"conds={p['n_conditions']}, status={p['validation_status']}")

    result = {
        'n_total_predictions': len(predictions),
        'n_confirmed': sum(1 for p in predictions if p['validation_status'] == 'confirmed'),
        'n_plausible': sum(1 for p in predictions if p['validation_status'] == 'plausible'),
        'n_novel': sum(1 for p in predictions if p['validation_status'] == 'novel'),
        'n_in_string': n_string,
        'n_in_trrust': n_trrust,
        'n_consensus': sum(1 for p in predictions if p['is_consensus']),
        'n_novel_domain': sum(1 for p in predictions if p['is_novel']),
        'top_predictions': predictions[:500],  # Save top 500
        'all_predictions_count': len(predictions),
        'validation_rates': {
            'confirmed_frac': round(sum(1 for p in predictions if p['validation_status'] == 'confirmed') / len(predictions), 4) if predictions else 0,
            'plausible_frac': round(sum(1 for p in predictions if p['validation_status'] == 'plausible') / len(predictions), 4) if predictions else 0,
            'novel_frac': round(sum(1 for p in predictions if p['validation_status'] == 'novel') / len(predictions), 4) if predictions else 0,
        },
    }

    # Also save a compact version of ALL predictions for Step 2
    # (just gene pairs + mean_d + edges + consensus flag)
    all_compact = []
    for p in predictions:
        all_compact.append({
            'sg': p['source_gene'],
            'tg': p['target_gene'],
            'md': p['mean_d'],
            'mad': p['mean_abs_d'],
            'ne': p['n_edges'],
            'cons': p['is_consensus'],
            'nov': p['is_novel'],
            'vs': p['validation_status'],
        })
    result['all_predictions_compact'] = all_compact

    elapsed = time.time() - t0
    result['runtime_seconds'] = round(elapsed, 1)
    print(f"\n  Step 1 complete in {elapsed:.0f}s")

    save_json(result, out_path)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Step 2: Perturbation Response Validation
# ════════════════════════════════════════════════════════════════════════════════
def step2_perturbation_validation(step1_result):
    out_path = os.path.join(OUT_DIR, "step2_perturbation_validation.json")
    if os.path.exists(out_path):
        print("Step 2 already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("\n" + "=" * 70)
    print("STEP 2: Perturbation Response Validation")
    print("=" * 70)
    t0 = time.time()

    # Load Replogle CRISPRi data
    print("Loading Replogle CRISPRi data...")
    import h5py
    h5 = h5py.File(REPLOGLE_PATH, 'r')

    # Get gene names (var_names)
    var_names_raw = h5['var']['gene_name_index'][:]
    var_names = [x.decode() if isinstance(x, bytes) else str(x) for x in var_names_raw]
    n_genes = len(var_names)
    print(f"  {n_genes} genes in expression matrix")

    # Get perturbation targets (obs.gene - categorical)
    obs_gene_codes = h5['obs']['gene']['codes'][:]
    obs_gene_cats = h5['obs']['gene']['categories'][:]
    obs_gene_cats = [x.decode() if isinstance(x, bytes) else str(x) for x in obs_gene_cats]
    pert_labels = [obs_gene_cats[c] for c in obs_gene_codes]
    n_cells = len(pert_labels)
    print(f"  {n_cells} cells, {len(set(pert_labels))} unique perturbation targets")

    # Identify control cells
    # Controls typically labeled as "non-targeting" or similar
    unique_perts = Counter(pert_labels)
    # Find control label - look for "non-targeting", "control", or most common
    control_label = None
    for candidate in ['non-targeting', 'Non-Targeting', 'control', 'ctrl', 'CTRL']:
        if candidate in unique_perts:
            control_label = candidate
            break
    if control_label is None:
        # Use the most common label (likely control)
        control_label = unique_perts.most_common(1)[0][0]
    print(f"  Control label: '{control_label}' ({unique_perts[control_label]} cells)")

    # Get control cell indices
    ctrl_idx = [i for i, l in enumerate(pert_labels) if l == control_label]

    # Load expression matrix (dense, float32)
    print("  Computing control mean expression...")
    X = h5['X']

    # Compute control mean in chunks to avoid memory issues
    chunk_size = 10000
    ctrl_sum = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, len(ctrl_idx), chunk_size):
        batch_idx = sorted(ctrl_idx[start:start + chunk_size])
        batch = X[batch_idx]
        ctrl_sum += batch.sum(axis=0)
    ctrl_mean = ctrl_sum / len(ctrl_idx)
    print(f"  Control mean computed from {len(ctrl_idx)} cells")

    # For each perturbation target, compute pseudobulk LFC vs control
    print("  Computing per-target LFC...")
    pert_targets = sorted(set(pert_labels) - {control_label})
    target_to_idx = defaultdict(list)
    for i, l in enumerate(pert_labels):
        if l != control_label:
            target_to_idx[l].append(i)

    # Build LFC matrix: target → gene → LFC
    lfc_dict = {}
    for target in pert_targets:
        idx = sorted(target_to_idx[target])
        if len(idx) < 5:
            continue
        # Compute mean in chunks
        target_sum = np.zeros(n_genes, dtype=np.float64)
        for start in range(0, len(idx), chunk_size):
            batch_idx = idx[start:start + chunk_size]
            batch = X[batch_idx]
            target_sum += batch.sum(axis=0)
        target_mean = target_sum / len(idx)

        # LFC = log2(target_mean + 1) - log2(ctrl_mean + 1)
        lfc = np.log2(target_mean + 1) - np.log2(ctrl_mean + 1)
        lfc_dict[target] = lfc

    h5.close()

    print(f"  LFC computed for {len(lfc_dict)} perturbation targets")

    # Build gene name → index mapping
    gene_to_idx = {g: i for i, g in enumerate(var_names)}

    # Load circuit predictions from Step 1
    predictions = step1_result['all_predictions_compact']
    print(f"  {len(predictions)} gene-pair predictions to validate")

    # Build source_gene → list of (target_gene, mean_d, is_consensus, is_novel)
    source_to_targets = defaultdict(list)
    for p in predictions:
        source_to_targets[p['sg']].append(p)

    # Find overlap: perturbation targets that are also source genes
    overlap_genes = set(lfc_dict.keys()) & set(source_to_targets.keys())
    print(f"  {len(overlap_genes)} perturbation targets overlap with circuit source genes")

    if len(overlap_genes) == 0:
        print("  WARNING: No overlap — cannot validate")
        result = {'n_overlap': 0, 'error': 'no overlap between perturbation targets and circuit source genes'}
        save_json(result, out_path)
        return result

    # Evaluation 1: Directional accuracy
    # For each (source_gene KD, target_gene):
    #   circuit predicts direction via mean_d sign
    #   actual direction from LFC
    print("\n  Evaluation 1: Directional accuracy...")
    concordant = 0
    discordant = 0
    n_tested = 0
    all_pairs_data = []

    for source_gene in overlap_genes:
        lfc = lfc_dict[source_gene]
        for pred in source_to_targets[source_gene]:
            tg = pred['tg']
            if tg not in gene_to_idx:
                continue
            tg_idx = gene_to_idx[tg]
            actual_lfc = lfc[tg_idx]
            predicted_d = pred['md']  # mean_d (signed)

            if abs(actual_lfc) < 0.01:  # Skip near-zero changes
                continue

            n_tested += 1
            # KD of source gene → if circuit edge is inhibitory (negative d),
            # removing inhibition should UPregulate target (positive LFC)
            # if edge is excitatory (positive d), removing activation should
            # DOWNregulate target (negative LFC)
            # So concordant = sign(d) != sign(LFC) actually...
            # Wait: KD = loss of source gene activity
            # If source→target is excitatory (d>0), KD should reduce target → LFC<0
            # If source→target is inhibitory (d<0), KD should increase target → LFC>0
            # Concordant: sign(d) * sign(LFC) < 0 (opposite signs)
            # Actually more nuanced: d is the effect of ABLATING the source feature
            # So d<0 means ablation decreases target, meaning source normally activates target
            # KD of source gene → same direction as ablation → expect LFC same sign as d
            # Actually let me think again:
            # cohens_d = (activation_with_ablation - activation_without_ablation) / pooled_sd
            # So d<0 means ablation DECREASES target activation
            # CRISPRi KD = reduces source gene expression ≈ ablation
            # So we expect LFC to have SAME sign as d
            same_sign = (predicted_d > 0 and actual_lfc > 0) or (predicted_d < 0 and actual_lfc < 0)
            if same_sign:
                concordant += 1
            else:
                discordant += 1

            all_pairs_data.append({
                'source': source_gene,
                'target': tg,
                'predicted_d': round(predicted_d, 4),
                'actual_lfc': round(float(actual_lfc), 4),
                'concordant': same_sign,
                'is_consensus': pred['cons'],
                'is_novel': pred['nov'],
            })

    sign_accuracy = concordant / n_tested if n_tested > 0 else 0
    print(f"    Tested: {n_tested} gene pairs")
    print(f"    Concordant: {concordant} ({sign_accuracy:.1%})")
    print(f"    Discordant: {discordant} ({1 - sign_accuracy:.1%})")

    # Evaluation 2: Magnitude correlation
    print("\n  Evaluation 2: Magnitude correlation...")
    predicted_abs_d = [abs(p['predicted_d']) for p in all_pairs_data]
    actual_abs_lfc = [abs(p['actual_lfc']) for p in all_pairs_data]

    if len(predicted_abs_d) >= 10:
        rho, p_val = stats.spearmanr(predicted_abs_d, actual_abs_lfc)
        print(f"    Spearman rho = {rho:.4f}, p = {p_val:.4e}")
    else:
        rho, p_val = 0.0, 1.0
        print(f"    Too few pairs ({len(predicted_abs_d)}) for correlation")

    # Evaluation 3: Enrichment
    # Among predicted target genes for each KD source, are responsive genes enriched?
    print("\n  Evaluation 3: Target enrichment...")
    LFC_THRESHOLD = 0.5
    enrichment_results = []

    for source_gene in overlap_genes:
        lfc = lfc_dict[source_gene]
        predicted_targets = set()
        for pred in source_to_targets[source_gene]:
            if pred['tg'] in gene_to_idx:
                predicted_targets.add(pred['tg'])

        if len(predicted_targets) < 3:
            continue

        # All genes in expression matrix
        all_genes_set = set(var_names)
        non_predicted = all_genes_set - predicted_targets - {source_gene}

        # Responsive genes (|LFC| > threshold)
        responsive = set()
        for g in var_names:
            idx = gene_to_idx[g]
            if abs(lfc[idx]) > LFC_THRESHOLD:
                responsive.add(g)

        # 2x2 contingency table
        a = len(predicted_targets & responsive)
        b = len(predicted_targets - responsive)
        c = len(non_predicted & responsive)
        d_val = len(non_predicted - responsive)

        if a + b > 0 and c + d_val > 0:
            odds_ratio, fisher_p = stats.fisher_exact([[a, b], [c, d_val]], alternative='greater')
            enrichment_results.append({
                'source_gene': source_gene,
                'n_predicted': len(predicted_targets),
                'n_responsive_in_predicted': a,
                'n_responsive_total': len(responsive),
                'odds_ratio': round(float(odds_ratio), 4),
                'p_value': float(fisher_p),
            })

    # Aggregate enrichment
    if enrichment_results:
        sig_enriched = sum(1 for e in enrichment_results if e['p_value'] < 0.05)
        finite_ors = [e['odds_ratio'] for e in enrichment_results if np.isfinite(e['odds_ratio'])]
        median_or = np.median(finite_ors) if finite_ors else 0.0
        enrichment_results.sort(key=lambda x: x['p_value'])
        print(f"    {len(enrichment_results)} source genes tested for target enrichment")
        print(f"    {sig_enriched} significantly enriched (p<0.05)")
        print(f"    Median odds ratio: {median_or:.2f}")
        print(f"    Top 5:")
        for e in enrichment_results[:5]:
            print(f"      {e['source_gene']}: OR={e['odds_ratio']:.2f}, p={e['p_value']:.4f}, "
                  f"{e['n_responsive_in_predicted']}/{e['n_predicted']} responsive")
    else:
        sig_enriched = 0
        median_or = 0

    # Evaluation 4: Consensus vs all — compare subsets
    print("\n  Evaluation 4: Consensus vs all comparison...")
    consensus_pairs_data = [p for p in all_pairs_data if p['is_consensus']]
    novel_pairs_data = [p for p in all_pairs_data if p['is_novel']]

    subsets = {
        'all': all_pairs_data,
        'consensus_only': consensus_pairs_data,
        'novel_only': novel_pairs_data,
    }
    subset_results = {}
    for name, subset in subsets.items():
        if len(subset) < 5:
            subset_results[name] = {'n': len(subset), 'sign_accuracy': None, 'rho': None}
            continue
        conc = sum(1 for p in subset if p['concordant'])
        acc = conc / len(subset)
        abs_d = [abs(p['predicted_d']) for p in subset]
        abs_l = [abs(p['actual_lfc']) for p in subset]
        r, pv = stats.spearmanr(abs_d, abs_l) if len(abs_d) >= 10 else (0, 1)
        subset_results[name] = {
            'n': len(subset),
            'sign_accuracy': round(acc, 4),
            'rho': round(float(r), 4),
            'rho_p': float(pv),
        }
        print(f"    {name}: n={len(subset)}, sign_accuracy={acc:.1%}, rho={r:.4f}")

    result = {
        'n_overlap_genes': len(overlap_genes),
        'n_perturbation_targets': len(lfc_dict),
        'n_tested_pairs': n_tested,
        'sign_accuracy': round(sign_accuracy, 4),
        'concordant': concordant,
        'discordant': discordant,
        'magnitude_correlation': {
            'spearman_rho': round(float(rho), 4),
            'p_value': float(p_val),
        },
        'target_enrichment': {
            'n_tested': len(enrichment_results),
            'n_significant': sig_enriched,
            'median_odds_ratio': round(float(median_or), 4),
            'top_results': enrichment_results[:20],
        },
        'subset_comparison': subset_results,
        'top_pairs': all_pairs_data[:200],  # Save top 200 by predicted |d|
        'runtime_seconds': round(time.time() - t0, 1),
    }

    print(f"\n  Step 2 complete in {time.time() - t0:.0f}s")
    save_json(result, out_path)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Step 3: Disease Gene Mapping
# ════════════════════════════════════════════════════════════════════════════════
def step3_disease_mapping():
    out_path = os.path.join(OUT_DIR, "step3_disease_mapping.json")
    if os.path.exists(out_path):
        print("Step 3 already complete, loading...")
        with open(out_path) as f:
            return json.load(f)

    print("\n" + "=" * 70)
    print("STEP 3: Disease Gene Mapping")
    print("=" * 70)
    t0 = time.time()

    # ── Define disease-relevant gene sets ──
    # Strategy: use GO BP terms as disease gene set proxies
    # Many GO terms directly map to disease-relevant processes
    print("Loading GO BP gene sets...")
    with open(os.path.join(BIO_DB, "go_bp_gene_sets.json")) as f:
        go_bp = json.load(f)

    # Define disease categories by keyword matching on GO BP terms
    disease_categories = {
        'DNA_damage_repair': ['dna repair', 'dna damage', 'dna integrity', 'double-strand break'],
        'cell_cycle_cancer': ['mitotic cell cycle', 'cell cycle checkpoint', 'cell division',
                              'chromosome segregation', 'sister chromatid'],
        'apoptosis': ['apoptotic process', 'programmed cell death', 'apoptotic signaling'],
        'immune_response': ['immune response', 'inflammatory response', 'interferon',
                           'cytokine', 'antigen', 'toll-like receptor', 'nf-kappab',
                           't cell activation', 'b cell activation'],
        'metabolism_cancer': ['glycolytic process', 'oxidative phosphorylation',
                             'cholesterol biosynthesis', 'fatty acid metabolic'],
        'transcription_regulation': ['regulation of transcription', 'chromatin remodeling',
                                    'histone modification', 'epigenetic'],
        'signaling_oncogenic': ['mapk cascade', 'ras protein signal', 'wnt signaling',
                               'notch signaling', 'hedgehog signaling', 'jak-stat',
                               'pi3k', 'mtor'],
        'protein_quality': ['unfolded protein response', 'er stress', 'protein folding',
                           'ubiquitin-dependent', 'proteasome', 'autophagy'],
        'angiogenesis': ['angiogenesis', 'blood vessel', 'vasculogenesis', 'vegf'],
        'metastasis': ['cell migration', 'cell adhesion', 'epithelial to mesenchymal',
                      'extracellular matrix', 'integrin'],
    }

    # Build disease gene sets from GO terms
    disease_gene_sets = {}
    disease_go_terms = {}
    for category, keywords in disease_categories.items():
        genes = set()
        matched_terms = []
        for term, term_genes in go_bp.items():
            term_lower = term.lower()
            if any(kw in term_lower for kw in keywords):
                genes.update(term_genes)
                matched_terms.append(term)
        disease_gene_sets[category] = genes
        disease_go_terms[category] = matched_terms
        print(f"  {category}: {len(genes)} genes from {len(matched_terms)} GO terms")

    # Also add TRRUST TFs as "transcriptional regulators"
    print("Loading TRRUST for TF classification...")
    trrust_tfs = set()
    trrust_targets = defaultdict(set)
    with open(TRRUST_PATH) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                trrust_tfs.add(parts[0])
                trrust_targets[parts[0]].add(parts[1])
    disease_gene_sets['trrust_TFs'] = trrust_tfs
    print(f"  trrust_TFs: {len(trrust_tfs)} transcription factors")

    # ── Load Phase 5 data ──
    print("\nLoading Phase 5 hierarchy data...")
    with open(os.path.join(P5_DIR, "step4_hierarchy.json")) as f:
        hierarchy = json.load(f)

    print("Loading annotated edges...")
    with open(os.path.join(P5_DIR, "step1_annotated_edges.json")) as f:
        all_edges = json.load(f)

    print("Loading consensus graph...")
    with open(os.path.join(P5_DIR, "step2_consensus_graph.json")) as f:
        consensus = json.load(f)
    consensus_pairs = set()
    for p in consensus['consensus_pairs']:
        consensus_pairs.add((p['source_domain'], p['target_domain']))

    # ── Collect all genes per domain (from circuit edges) ──
    print("Building domain → gene mappings from circuit edges...")
    domain_genes = defaultdict(set)
    for cond, edges in all_edges.items():
        for edge in edges:
            src_label = edge.get('source_label', 'unknown')
            tgt_label = edge.get('target_label', 'unannotated')
            for g in edge.get('source_genes', []):
                domain_genes[src_label].add(g)
            for g in edge.get('target_genes', []):
                domain_genes[tgt_label].add(g)

    # All genes in any feature (background)
    all_circuit_genes = set()
    for genes in domain_genes.values():
        all_circuit_genes.update(genes)
    n_background = len(all_circuit_genes)
    print(f"  {len(domain_genes)} domains, {n_background} unique genes")

    # ── Map disease gene sets to circuit domains ──
    print("\nMapping disease gene sets to circuit domains...")
    domain_disease_enrichment = {}

    for domain, domain_gene_set in domain_genes.items():
        if domain in ('unknown', 'unannotated') or len(domain_gene_set) < 3:
            continue
        enrichments = {}
        for disease, disease_genes in disease_gene_sets.items():
            overlap = domain_gene_set & disease_genes
            if len(overlap) < 1:
                continue
            # Fisher's exact test
            a = len(overlap)
            b = len(domain_gene_set - disease_genes)
            c = len(disease_genes - domain_gene_set)
            d_val = n_background - len(domain_gene_set | disease_genes)
            if d_val < 0:
                d_val = 0
            odds_ratio, p_val = stats.fisher_exact([[a, b], [c, d_val]], alternative='greater')
            if p_val < 0.05:
                enrichments[disease] = {
                    'odds_ratio': round(float(odds_ratio), 3),
                    'p_value': float(p_val),
                    'overlap_genes': sorted(overlap)[:20],
                    'n_overlap': len(overlap),
                }
        if enrichments:
            domain_disease_enrichment[domain] = enrichments

    print(f"  {len(domain_disease_enrichment)} domains enriched for ≥1 disease category")

    # ── Map to circuit hubs (PageRank top 50) ──
    print("\nMapping disease enrichment to circuit hubs...")
    hub_disease = {}
    for hub in hierarchy['pagerank_top50']:
        domain = hub['domain']
        if domain in domain_disease_enrichment:
            hub_disease[domain] = {
                'pagerank': hub['pagerank'],
                'in_degree': hub['in_degree'],
                'out_degree': hub['out_degree'],
                'disease_enrichments': domain_disease_enrichment[domain],
            }
    print(f"  {len(hub_disease)} of top-50 hubs enriched for disease gene sets")

    for domain, info in list(hub_disease.items())[:10]:
        diseases = ', '.join(info['disease_enrichments'].keys())
        print(f"    {domain.split(' (GO:')[0]}: {diseases}")

    # ── Are disease-associated domains more central? ──
    print("\nTesting whether disease domains are more central...")
    domain_layers = hierarchy['domain_mean_layers']

    # Get PageRank for all domains (approximate from layer stats)
    disease_domain_names = set(domain_disease_enrichment.keys())
    all_domain_names = set(domain_layers.keys())

    # Count edges per domain as centrality proxy
    domain_edge_counts = Counter()
    for cond, edges in all_edges.items():
        for edge in edges:
            domain_edge_counts[edge.get('source_label', 'unknown')] += 1
            domain_edge_counts[edge.get('target_label', 'unannotated')] += 1

    disease_centralities = []
    nondisease_centralities = []
    for domain in all_domain_names:
        if domain in ('unknown', 'unannotated'):
            continue
        ec = domain_edge_counts.get(domain, 0)
        if domain in disease_domain_names:
            disease_centralities.append(ec)
        else:
            nondisease_centralities.append(ec)

    if disease_centralities and nondisease_centralities:
        u_stat, u_p = stats.mannwhitneyu(disease_centralities, nondisease_centralities, alternative='greater')
        disease_median = np.median(disease_centralities)
        nondisease_median = np.median(nondisease_centralities)
        print(f"  Disease domains: median {disease_median:.0f} edges")
        print(f"  Non-disease: median {nondisease_median:.0f} edges")
        print(f"  Mann-Whitney U p = {u_p:.4e}")
    else:
        u_p = 1.0
        disease_median = 0
        nondisease_median = 0

    # ── Disease circuit paths ──
    print("\nIdentifying disease circuit paths...")
    disease_circuits = {}
    for disease_name, disease_genes in disease_gene_sets.items():
        # Find domains enriched for this disease
        enriched_domains = set()
        for domain, enrichments in domain_disease_enrichment.items():
            if disease_name in enrichments:
                enriched_domains.add(domain)

        if len(enriched_domains) < 2:
            disease_circuits[disease_name] = {'n_enriched_domains': len(enriched_domains), 'n_circuit_edges': 0}
            continue

        # Find circuit edges between enriched domains
        circuit_edges = []
        for cond, edges in all_edges.items():
            for edge in edges:
                src = edge.get('source_label', 'unknown')
                tgt = edge.get('target_label', 'unannotated')
                if src in enriched_domains and tgt in enriched_domains:
                    circuit_edges.append({
                        'condition': cond,
                        'source_domain': src,
                        'target_domain': tgt,
                        'cohens_d': edge['cohens_d'],
                        'source_layer': edge['source_layer'],
                        'target_layer': edge['target_layer'],
                    })

        # Check consensus overlap
        n_consensus = 0
        circuit_domain_pairs = set()
        for e in circuit_edges:
            pair = (e['source_domain'], e['target_domain'])
            circuit_domain_pairs.add(pair)
            if pair in consensus_pairs:
                n_consensus += 1

        mean_abs_d = np.mean([abs(e['cohens_d']) for e in circuit_edges]) if circuit_edges else 0
        mean_layer_span = np.mean([abs(e['target_layer'] - e['source_layer']) for e in circuit_edges]) if circuit_edges else 0

        disease_circuits[disease_name] = {
            'n_enriched_domains': len(enriched_domains),
            'n_circuit_edges': len(circuit_edges),
            'n_unique_domain_pairs': len(circuit_domain_pairs),
            'n_consensus_edges': n_consensus,
            'mean_abs_d': round(float(mean_abs_d), 3),
            'mean_layer_span': round(float(mean_layer_span), 1),
            'top_edges': sorted(circuit_edges, key=lambda x: abs(x['cohens_d']), reverse=True)[:10],
        }
        if circuit_edges:
            print(f"  {disease_name}: {len(enriched_domains)} domains, "
                  f"{len(circuit_edges)} edges, {n_consensus} consensus, "
                  f"mean |d|={mean_abs_d:.2f}")

    # ── Cross-model disease validation ──
    print("\nCross-model disease validation...")
    # Are disease circuit edges more likely to be in consensus?
    all_domain_pairs_in_circuits = set()
    disease_domain_pairs_in_circuits = set()
    for cond, edges in all_edges.items():
        for edge in edges:
            src = edge.get('source_label', 'unknown')
            tgt = edge.get('target_label', 'unannotated')
            pair = (src, tgt)
            all_domain_pairs_in_circuits.add(pair)
            # Is this a disease circuit edge?
            for disease_name, dc in disease_circuits.items():
                if dc['n_circuit_edges'] > 0:
                    enriched_doms_for_disease = set()
                    for dom, enrichments in domain_disease_enrichment.items():
                        if disease_name in enrichments:
                            enriched_doms_for_disease.add(dom)
                    if src in enriched_doms_for_disease and tgt in enriched_doms_for_disease:
                        disease_domain_pairs_in_circuits.add(pair)
                        break

    disease_in_consensus = disease_domain_pairs_in_circuits & consensus_pairs
    nondisease_pairs = all_domain_pairs_in_circuits - disease_domain_pairs_in_circuits
    nondisease_in_consensus = nondisease_pairs & consensus_pairs

    a = len(disease_in_consensus)
    b = len(disease_domain_pairs_in_circuits) - a
    c = len(nondisease_in_consensus)
    d_val = len(nondisease_pairs) - c

    if a + b > 0 and c + d_val > 0:
        cross_or, cross_p = stats.fisher_exact([[a, b], [c, d_val]], alternative='greater')
        print(f"  Disease pairs in consensus: {a}/{a+b} ({a/(a+b)*100:.1f}%)")
        print(f"  Non-disease in consensus: {c}/{c+d_val} ({c/(c+d_val)*100:.1f}%)")
        print(f"  Fisher's OR={cross_or:.2f}, p={cross_p:.4f}")
    else:
        cross_or, cross_p = 1.0, 1.0

    result = {
        'disease_categories': {k: {'n_genes': len(v), 'n_go_terms': len(disease_go_terms.get(k, []))}
                               for k, v in disease_gene_sets.items()},
        'n_domains_enriched': len(domain_disease_enrichment),
        'hub_disease_mapping': hub_disease,
        'n_hubs_with_disease': len(hub_disease),
        'centrality_test': {
            'disease_median_edges': round(float(disease_median), 1),
            'nondisease_median_edges': round(float(nondisease_median), 1),
            'mann_whitney_p': float(u_p),
        },
        'disease_circuits': disease_circuits,
        'cross_model_validation': {
            'disease_consensus_frac': round(a / (a + b), 4) if a + b > 0 else 0,
            'nondisease_consensus_frac': round(c / (c + d_val), 4) if c + d_val > 0 else 0,
            'odds_ratio': round(float(cross_or), 3),
            'p_value': float(cross_p),
        },
        # Top enriched domains per disease (for paper)
        'top_domain_enrichments': {},
        'runtime_seconds': round(time.time() - t0, 1),
    }

    # Add top enriched domains per disease category
    for disease_name in disease_gene_sets:
        top_doms = []
        for domain, enrichments in domain_disease_enrichment.items():
            if disease_name in enrichments:
                top_doms.append({
                    'domain': domain,
                    'odds_ratio': enrichments[disease_name]['odds_ratio'],
                    'p_value': enrichments[disease_name]['p_value'],
                    'n_overlap': enrichments[disease_name]['n_overlap'],
                })
        top_doms.sort(key=lambda x: x['p_value'])
        result['top_domain_enrichments'][disease_name] = top_doms[:10]

    print(f"\n  Step 3 complete in {time.time() - t0:.0f}s")
    save_json(result, out_path)
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Phase 6: Gene-Level Predictions, Perturbation Validation & Disease Mapping")
    print("=" * 70)
    t_total = time.time()

    # Step 1: Gene-level predictions
    s1 = step1_gene_predictions()

    # Step 2: Perturbation validation (uses Step 1 results)
    s2 = step2_perturbation_validation(s1)

    # Step 3: Disease gene mapping (independent of Steps 1-2)
    s3 = step3_disease_mapping()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"Phase 6 COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Step 1: {s1.get('n_total_predictions', '?')} gene-pair predictions")
    print(f"  Step 2: Sign accuracy = {s2.get('sign_accuracy', '?')}")
    print(f"  Step 3: {s3.get('n_domains_enriched', '?')} disease-enriched domains")
    print(f"Output: {OUT_DIR}")
