#!/usr/bin/env python3
"""
Phase 4: Causal Feature-to-Feature Circuit Tracing (Geneformer).

Ablate individual SAE features at source layers and measure how downstream
SAE features change — revealing directed information flow through the model.

For each source feature:
  1. Run clean forward pass → capture hidden states at ALL downstream layers
  2. Encode downstream hidden states through downstream SAEs → clean activations
  3. Ablate source feature: SAE encode → zero feature → decode → compute delta
  4. Hook-based ablated forward pass → capture downstream hidden states
  5. Encode ablated hidden states through downstream SAEs → ablated activations
  6. Accumulate per-downstream-feature deltas across cells (Welford's algorithm)
  7. Compute significance: Cohen's d + consistency

Prerequisites (from bio-sae, Phases 1-3):
  - Trained SAE checkpoints: experiments/{phase}/sae_models/layer{NN}_x4_k32/sae_final.pt
  - Feature annotations: experiments/{phase}/sae_models/layer{NN}_x4_k32/feature_annotations.json
  - Geneformer V2-316M model (HuggingFace: ctheodoris/Geneformer)
  - Replogle CRISPRi K562 data (for cell tokenization)

Configuration:
  Set BASE below to your local root directory containing the bio-sae experiment outputs.
  Adjust PROJ_DIR, DATA_DIR, SAE_BASE, DATA_PATH as needed for your directory layout.

Usage:
    python src/13_causal_circuit_tracing.py \
        [--source-layers 0,5,11,15] [--n-features 30] [--n-cells 200] \
        [--sae-dir path/to/sae_models] [--available-layers 0,1,...,17] \
        [--data-source k562|ts]
"""

import os
import sys
import gc
import json
import time
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

import numpy as np

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PAPER_DIR = os.path.join(BASE, "biodyn-nmi-paper")
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
DATA_DIR = os.path.join(PROJ_DIR, "experiments/phase1_k562")
SAE_BASE = os.path.join(DATA_DIR, "sae_models")
TOKEN_DICTS_DIR = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data")
DATA_PATH = os.path.join(PAPER_DIR, "src/02_cssi_method/crispri_validation/data/replogle_concat.h5ad")

MODEL_NAME = "ctheodoris/Geneformer"
MODEL_SUBFOLDER = "Geneformer-V2-316M"

EXPANSION = 4
K_VAL = 32
N_CTRL = 2000
MAX_SEQ_LEN = 2048
HIDDEN_DIM = 1152
N_FEATURES = 4608  # HIDDEN_DIM * EXPANSION
N_LAYERS = 18

# Significance thresholds
COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7
TOP_EFFECTS_PER_LAYER = 50  # Keep top N effects per downstream layer in output


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_categorical_column(h5group, col_name):
    import h5py
    col = h5group[col_name]
    if isinstance(col, h5py.Group):
        categories = col['categories'][:]
        codes = col['codes'][:]
        if categories.dtype.kind in ('O', 'S'):
            categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
        return categories[codes]
    else:
        data = col[:]
        if data.dtype.kind in ('O', 'S'):
            return np.array([x.decode() if isinstance(x, bytes) else x for x in data])
        return data


def tokenize_cell(expression_vector, var_indices, token_ids, medians, max_len=2048):
    expr = expression_vector[var_indices]
    nonzero = expr > 0
    if nonzero.sum() == 0:
        return None
    expr_nz = expr[nonzero]
    tokens_nz = token_ids[nonzero]
    medians_nz = medians[nonzero]
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = expr_nz / medians_nz
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0)
    rank_order = np.argsort(-normalized)
    ranked_tokens = tokens_nz[rank_order][:max_len - 2]
    return np.concatenate([[2], ranked_tokens, [3]]).astype(np.int64)


def select_features(layer, n_features=30):
    """Select well-annotated features for circuit tracing.

    Reuses the scoring from 08_causal_patching.py:
    score = n_ontologies * 10 + n_annotations - log10(min_p)
    """
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)

    with open(os.path.join(run_dir, "feature_annotations.json")) as f:
        ann_data = json.load(f)
    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)

    feature_annotations = ann_data.get('feature_annotations', {})

    # Build gene sets and freq for each feature
    feature_genes = {}
    feature_freq = {}
    for feat in catalog['features']:
        fi = feat['feature_idx']
        if feat.get('top_genes'):
            feature_genes[fi] = [g['gene_name'] for g in feat['top_genes'][:20]]
        feature_freq[fi] = feat.get('activation_freq', 0)

    # Score features by annotation quality
    scored = []
    for fid_str, anns in feature_annotations.items():
        fi = int(fid_str)
        if fi not in feature_genes or len(feature_genes[fi]) < 10:
            continue
        if feature_freq.get(fi, 0) < 0.01:
            continue

        ontologies = set(a['ontology'] for a in anns)
        n_ont = len(ontologies)
        n_ann = len(anns)
        min_p = min(a.get('p_adjusted', 1.0) for a in anns) if anns else 1.0

        best_label = "unknown"
        for a in anns:
            if a['ontology'] in ('GO_BP', 'KEGG', 'Reactome'):
                best_label = a['term']
                break

        scored.append({
            'feature_idx': fi,
            'n_ontologies': n_ont,
            'n_annotations': n_ann,
            'min_p': min_p,
            'label': best_label,
            'top_genes': feature_genes[fi],
            'activation_freq': feature_freq.get(fi, 0),
            'score': n_ont * 10 + n_ann - np.log10(max(min_p, 1e-30)),
        })

    scored.sort(key=lambda x: -x['score'])
    selected = scored[:n_features]
    print(f"  Selected {len(selected)} features for circuit tracing at layer {layer}")
    for i, s in enumerate(selected[:10]):
        print(f"    [{i}] Feature {s['feature_idx']}: {s['n_ontologies']} ont, "
              f"{s['n_annotations']} ann, freq={s['activation_freq']:.3f} | {s['label'][:50]}")

    return selected


class WelfordAccumulator:
    """Online mean/variance accumulator using Welford's algorithm."""

    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)
        self.pos_count = np.zeros(n_features, dtype=np.float64)

    def update(self, delta_vector):
        """Update with a single observation vector."""
        self.n += 1
        d = delta_vector - self.mean
        self.mean += d / self.n
        d2 = delta_vector - self.mean
        self.M2 += d * d2
        self.pos_count += (delta_vector > 0).astype(np.float64)

    def finalize(self):
        """Compute Cohen's d and consistency for all features."""
        if self.n < 2:
            return (np.zeros_like(self.mean), np.zeros_like(self.mean))
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var)
        cohens_d = self.mean / np.maximum(std, 1e-10)
        consistency = np.maximum(self.pos_count, self.n - self.pos_count) / self.n
        return cohens_d, consistency


class SAECache:
    """Lazy-loading cache for downstream SAE models."""

    def __init__(self):
        self._saes = {}
        self._means = {}

    def get(self, layer):
        """Get SAE and activation mean for a layer."""
        if layer not in self._saes:
            import torch
            sys.path.insert(0, os.path.dirname(__file__))
            from sae_model import TopKSAE

            run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
            run_dir = os.path.join(SAE_BASE, run_name)
            sae = TopKSAE.load(os.path.join(run_dir, "sae_final.pt"), device='cpu')
            sae.eval()
            act_mean = np.load(os.path.join(run_dir, "activation_mean.npy"))
            self._saes[layer] = sae
            self._means[layer] = act_mean
            print(f"    Loaded SAE for layer {layer}")

        return self._saes[layer], self._means[layer]

    def preload(self, layers):
        """Preload SAEs for given layers."""
        for l in layers:
            self.get(l)


def load_sparse_row(f_group, row_idx, n_cols):
    """Load a single row from a sparse CSR matrix in h5py."""
    indptr = f_group['indptr']
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    indices = f_group['indices'][start:end]
    data = f_group['data'][start:end]
    row = np.zeros(n_cols, dtype=np.float32)
    row[indices] = data
    return row


def load_and_tokenize_cells(n_cells, data_source='k562'):
    """Load and tokenize cells. Returns list of token arrays.

    data_source: 'k562' for Replogle CRISPRi controls, or 'tabula_sapiens'
    for Tabula Sapiens multi-tissue cells (immune+kidney+lung).
    """
    import h5py

    print("  Loading tokenizer dictionaries...")
    with open(os.path.join(TOKEN_DICTS_DIR, "token_dictionary_gc104M.pkl"), 'rb') as f:
        token_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_median_dictionary_gc104M.pkl"), 'rb') as f:
        gene_median_dict = pickle.load(f)
    with open(os.path.join(TOKEN_DICTS_DIR, "gene_name_id_dict_gc104M.pkl"), 'rb') as f:
        gene_name_id_dict = pickle.load(f)

    if data_source == 'tabula_sapiens':
        return _load_tabula_sapiens_cells(n_cells, token_dict, gene_median_dict, h5py)
    else:
        return _load_k562_cells(n_cells, token_dict, gene_median_dict,
                                gene_name_id_dict, h5py)


def _load_k562_cells(n_cells, token_dict, gene_median_dict, gene_name_id_dict, h5py):
    """Load K562 CRISPRi control cells."""
    print("  Loading K562 cell data...")
    with h5py.File(DATA_PATH, 'r') as f:
        cell_genes = load_categorical_column(f['obs'], 'gene')
        var_genes = load_categorical_column(f['var'], 'gene_name_index')
        n_cells_total, n_genes_total = f['X'].shape

    control_mask = np.zeros(n_cells_total, dtype=bool)
    for ctrl_name in ['non-targeting', 'Non-targeting', 'non_targeting']:
        control_mask |= (cell_genes == ctrl_name)
    ctrl_indices = np.where(control_mask)[0]

    np.random.seed(42)
    if len(ctrl_indices) > N_CTRL:
        ctrl_indices = np.random.choice(ctrl_indices, N_CTRL, replace=False)
        ctrl_indices.sort()

    cell_sample = ctrl_indices[:n_cells]
    print(f"    Using {len(cell_sample)} K562 control cells")

    # Load expression
    with h5py.File(DATA_PATH, 'r') as f:
        X_sample = np.empty((len(cell_sample), n_genes_total), dtype=np.float32)
        for ci, idx in enumerate(cell_sample):
            X_sample[ci, :] = f['X'][int(idx), :]

    row_sums = X_sample.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_sample = np.log1p(X_sample / row_sums * 1e4)

    # Build tokenization arrays
    mapped_var_indices = []
    mapped_token_ids_list = []
    mapped_medians_list = []
    for i in range(n_genes_total):
        gene_name = var_genes[i]
        ensembl = gene_name_id_dict.get(gene_name)
        if ensembl and ensembl in token_dict:
            mapped_var_indices.append(i)
            mapped_token_ids_list.append(token_dict[ensembl])
            mapped_medians_list.append(gene_median_dict.get(ensembl, 1.0))
    mapped_var_indices = np.array(mapped_var_indices)
    mapped_token_ids = np.array(mapped_token_ids_list)
    mapped_medians = np.array(mapped_medians_list)

    # Tokenize
    all_tokens = []
    for ci in range(len(cell_sample)):
        tokens = tokenize_cell(X_sample[ci], mapped_var_indices,
                               mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
        if tokens is not None:
            all_tokens.append(tokens)
    print(f"    Tokenized {len(all_tokens)} cells")
    del X_sample
    gc.collect()

    return all_tokens


TS_TISSUES = {
    'immune': os.path.join(BASE, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"),
    'kidney': os.path.join(BASE, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_kidney.h5ad"),
    'lung': os.path.join(BASE, "biodyn-work/single_cell_mechinterp/data/raw/tabula_sapiens_lung.h5ad"),
}


def _load_tabula_sapiens_cells(n_cells, token_dict, gene_median_dict, h5py):
    """Load and tokenize Tabula Sapiens cells from immune+kidney+lung tissues.

    Allocates n_cells evenly across 3 tissues with stratified cell-type sampling.
    """
    print("  Loading Tabula Sapiens cells...")
    per_tissue = n_cells // 3
    remainder = n_cells - per_tissue * 3

    all_tokens = []
    rng = np.random.RandomState(42)

    for ti, (tissue_name, h5_path) in enumerate(TS_TISSUES.items()):
        n_this = per_tissue + (1 if ti < remainder else 0)

        # Select cells with stratified sampling
        with h5py.File(h5_path, 'r') as f:
            cell_types = load_categorical_column(f['obs'], 'cell_type')
            n_total = len(cell_types)
            n_genes = len(f['var']['_index'])

        n_this = min(n_this, n_total)
        unique_types, type_counts = np.unique(cell_types, return_counts=True)

        # Proportional allocation with minimum 1 per type
        allocations = {}
        remaining = n_this
        for ct, count in sorted(zip(unique_types, type_counts), key=lambda x: x[1]):
            alloc = max(1, int(round(count / n_total * n_this)))
            alloc = min(alloc, count, remaining)
            allocations[ct] = alloc
            remaining -= alloc
            if remaining <= 0:
                break
        if remaining > 0:
            for ct, count in sorted(zip(unique_types, type_counts), key=lambda x: -x[1]):
                can_add = min(remaining, count - allocations.get(ct, 0))
                if can_add > 0:
                    allocations[ct] = allocations.get(ct, 0) + can_add
                    remaining -= can_add
                if remaining <= 0:
                    break

        selected_indices = []
        for ct, n_select in allocations.items():
            ct_idx = np.where(cell_types == ct)[0]
            chosen = rng.choice(ct_idx, min(n_select, len(ct_idx)), replace=False)
            selected_indices.extend(chosen.tolist())
        selected_indices.sort()

        # Build gene mapping (Ensembl ID → token dict)
        with h5py.File(h5_path, 'r') as f:
            var_index = f['var']['_index'][:]

        mapped_var_indices = []
        mapped_token_ids_list = []
        mapped_medians_list = []
        for i in range(n_genes):
            ens_id = var_index[i].decode() if isinstance(var_index[i], bytes) else var_index[i]
            if ens_id in token_dict:
                mapped_var_indices.append(i)
                mapped_token_ids_list.append(token_dict[ens_id])
                mapped_medians_list.append(gene_median_dict.get(ens_id, 1.0))
        mapped_var_indices = np.array(mapped_var_indices)
        mapped_token_ids = np.array(mapped_token_ids_list)
        mapped_medians = np.array(mapped_medians_list)

        # Tokenize each cell (sparse row loading)
        tissue_tokens = 0
        with h5py.File(h5_path, 'r') as f:
            for cell_idx in selected_indices:
                expr = load_sparse_row(f['X'], cell_idx, n_genes)
                row_sum = expr.sum()
                if row_sum > 0:
                    expr = np.log1p(expr / row_sum * 1e4)
                tokens = tokenize_cell(expr, mapped_var_indices,
                                       mapped_token_ids, mapped_medians, MAX_SEQ_LEN)
                if tokens is not None:
                    all_tokens.append(tokens)
                    tissue_tokens += 1

        n_types_used = len(allocations)
        print(f"    {tissue_name}: {tissue_tokens} cells tokenized "
              f"({len(mapped_var_indices)}/{n_genes} genes mapped, "
              f"{n_types_used} cell types)")

    print(f"    Total: {len(all_tokens)} Tabula Sapiens cells tokenized")
    return all_tokens


def make_hook(modified_hidden, device):
    """Create a hook that replaces hidden state at the target layer."""
    fired = [False]
    def hook_fn(module, input, output):
        if fired[0]:
            return output
        fired[0] = True
        modified = list(output)
        modified[0] = modified_hidden.unsqueeze(0).to(device)
        return tuple(modified)
    return hook_fn


def trace_source_layer(source_layer, selected_features, all_tokens, model, device,
                       sae_cache, out_dir, n_cells, available_layers=None):
    """Run circuit tracing for all source features at one source layer.

    OPTIMIZED: Iterates over cells in the outer loop. For each cell, runs ONE
    clean forward pass, then processes all features that are active in that cell.
    This avoids redundant clean forward passes (30x fewer total passes).

    If available_layers is provided, downstream tracing is restricted to only
    those layers (for multi-tissue SAEs that exist at a subset of layers).
    """
    import torch

    partial_path = os.path.join(out_dir, f"circuit_tracing_partial_L{source_layer:02d}.json")
    final_path = os.path.join(out_dir, f"circuit_L{source_layer:02d}_features.json")

    if os.path.exists(final_path):
        print(f"  Already done: {final_path}")
        return final_path

    # Load source SAE
    source_sae, source_act_mean = sae_cache.get(source_layer)
    source_act_mean_t = torch.tensor(source_act_mean, dtype=torch.float32)

    # Determine downstream layers
    if available_layers is not None:
        downstream_layers = sorted([l for l in available_layers if l > source_layer])
    else:
        downstream_layers = list(range(source_layer + 1, N_LAYERS))
    print(f"  Source layer {source_layer} → tracing through {len(downstream_layers)} downstream layers: {downstream_layers}")

    # Preload downstream SAEs and cache their mean tensors
    sae_cache.preload(downstream_layers)
    dst_act_mean_tensors = {}
    for dl in downstream_layers:
        _, dst_mean = sae_cache.get(dl)
        dst_act_mean_tensors[dl] = torch.tensor(dst_mean, dtype=torch.float32)

    # Build feature index: fi → index into selected_features
    feature_indices = [sf['feature_idx'] for sf in selected_features]
    fi_to_idx = {fi: idx for idx, fi in enumerate(feature_indices)}

    # Initialize Welford accumulators: one per (feature, downstream_layer)
    # accumulators[fi][dl] = WelfordAccumulator
    accumulators = {}
    n_cells_active = {}  # per feature
    for sf in selected_features:
        fi = sf['feature_idx']
        accumulators[fi] = {dl: WelfordAccumulator(N_FEATURES) for dl in downstream_layers}
        n_cells_active[fi] = 0

    # Check for partial results (resume at cell level)
    start_cell = 0
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        start_cell = partial.get('cells_completed', 0)
        # Restore accumulators from partial
        for feat_data in partial.get('accumulator_state', []):
            fi = feat_data['feature_idx']
            if fi not in accumulators:
                continue
            n_cells_active[fi] = feat_data.get('n_cells_active', 0)
            for dl_str, acc_state in feat_data.get('accumulators', {}).items():
                dl = int(dl_str)
                if dl in accumulators[fi]:
                    acc = accumulators[fi][dl]
                    acc.n = acc_state['n']
                    acc.mean = np.array(acc_state['mean'])
                    acc.M2 = np.array(acc_state['M2'])
                    acc.pos_count = np.array(acc_state['pos_count'])
        print(f"  Resuming from cell {start_cell}/{n_cells}")

    total_t0 = time.time()
    n_ablated_passes = 0
    n_clean_passes = 0

    cells_to_process = all_tokens[:n_cells]
    print(f"  Processing {len(cells_to_process)} cells (starting from {start_cell})...")

    for ci in range(start_cell, len(cells_to_process)):
        tokens = cells_to_process[ci]
        seq_len = len(tokens)
        gene_mask = (tokens != 2) & (tokens != 3)
        gene_positions = np.where(gene_mask)[0]

        if len(gene_positions) == 0:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)

        # === ONE clean forward pass per cell ===
        with torch.no_grad():
            outputs_clean = model(input_ids=input_ids, attention_mask=attention_mask)
        n_clean_passes += 1

        # Get source layer hidden states at gene positions
        clean_hidden_full = outputs_clean.hidden_states[source_layer + 1][0].cpu()  # (seq_len, 1152)
        gene_hidden = clean_hidden_full[gene_positions]
        gene_hidden_centered = gene_hidden - source_act_mean_t

        # Encode through source SAE
        with torch.no_grad():
            h_sparse, topk_indices = source_sae.encode(gene_hidden_centered)

        # Pre-extract clean downstream hidden states at gene positions (CPU)
        clean_downstream = {}
        for dl in downstream_layers:
            clean_downstream[dl] = outputs_clean.hidden_states[dl + 1][0][gene_positions].cpu()

        # Check which features are active in this cell and process them
        for fi in feature_indices:
            active_mask = (topk_indices == fi).any(dim=1)  # (n_genes,)
            if not active_mask.any():
                continue

            n_cells_active[fi] += 1

            # --- Ablation: zero target feature ---
            h_ablated = h_sparse.clone()
            h_ablated[:, fi] = 0.0

            with torch.no_grad():
                recon_normal = source_sae.decode(h_sparse) + source_act_mean_t
                recon_ablated = source_sae.decode(h_ablated) + source_act_mean_t

            delta = recon_ablated - recon_normal

            # Build modified hidden state
            modified_hidden = clean_hidden_full.clone()
            modified_hidden[gene_positions] = clean_hidden_full[gene_positions] + delta

            # --- Ablated forward pass ---
            hook_handle = model.bert.encoder.layer[source_layer].register_forward_hook(
                make_hook(modified_hidden, device)
            )

            with torch.no_grad():
                outputs_ablated = model(input_ids=input_ids, attention_mask=attention_mask)

            hook_handle.remove()
            n_ablated_passes += 1

            # --- Encode downstream layers and accumulate ---
            # Average delta across active positions within this cell for Welford
            active_positions = torch.where(active_mask)[0]

            for dl in downstream_layers:
                dst_sae, _ = sae_cache.get(dl)
                dst_mean_t = dst_act_mean_tensors[dl]

                clean_dl = clean_downstream[dl]
                ablated_dl = outputs_ablated.hidden_states[dl + 1][0][gene_positions].cpu()

                with torch.no_grad():
                    clean_sparse, _ = dst_sae.encode(clean_dl - dst_mean_t)
                    ablated_sparse, _ = dst_sae.encode(ablated_dl - dst_mean_t)

                # Average delta across active positions → one observation per cell
                pos_deltas = []
                for pos in active_positions:
                    pos_deltas.append(
                        (ablated_sparse[pos] - clean_sparse[pos]).numpy().astype(np.float64)
                    )
                cell_delta = np.mean(pos_deltas, axis=0)
                accumulators[fi][dl].update(cell_delta)

            del outputs_ablated, modified_hidden, h_ablated, delta

        # Cleanup after processing all features for this cell
        del outputs_clean, clean_hidden_full, gene_hidden, h_sparse, topk_indices
        del clean_downstream
        if device.type == 'mps':
            torch.mps.empty_cache()

        # Progress report every 20 cells
        if (ci + 1) % 20 == 0:
            elapsed = time.time() - total_t0
            rate = (ci + 1 - start_cell) / elapsed
            remaining = (len(cells_to_process) - ci - 1) / max(rate, 0.001)
            print(f"    Cell {ci+1}/{len(cells_to_process)} | "
                  f"{n_clean_passes} clean + {n_ablated_passes} ablated passes | "
                  f"{rate:.2f} cells/s | ETA {remaining/60:.1f} min")

        # Incremental save every 50 cells
        if (ci + 1) % 50 == 0:
            _save_partial_v2(partial_path, source_layer, ci + 1, feature_indices,
                             accumulators, n_cells_active, n_cells)

    # === Finalize: compute significance and build results ===
    print(f"\n  Finalizing results for source layer {source_layer}...")
    feature_results = []

    for fi_idx, sf in enumerate(selected_features):
        fi = sf['feature_idx']

        downstream_effects = {}
        total_sig_edges = 0
        attenuation_curve = []

        for dl in downstream_layers:
            acc = accumulators[fi][dl]
            if acc.n < 2:
                downstream_effects[str(dl)] = {
                    'n_significant': 0,
                    'n_cells_measured': acc.n,
                    'top_effects': [],
                }
                attenuation_curve.append(0.0)
                continue

            cohens_d, consistency = acc.finalize()

            sig_mask = (np.abs(cohens_d) > COHENS_D_THRESHOLD) & (consistency > CONSISTENCY_THRESHOLD)
            n_sig = int(sig_mask.sum())
            total_sig_edges += n_sig

            sig_indices = np.where(sig_mask)[0]
            if len(sig_indices) > 0:
                abs_d = np.abs(cohens_d[sig_indices])
                top_order = np.argsort(-abs_d)[:TOP_EFFECTS_PER_LAYER]
                top_indices = sig_indices[top_order]

                std = np.sqrt(acc.M2 / (acc.n - 1))
                top_effects = []
                for idx in top_indices:
                    top_effects.append({
                        'target_feature_idx': int(idx),
                        'mean_delta': float(acc.mean[idx]),
                        'std_delta': float(std[idx]),
                        'cohens_d': float(cohens_d[idx]),
                        'consistency': float(consistency[idx]),
                        'n_cells_measured': int(acc.n),
                    })
            else:
                top_effects = []

            downstream_effects[str(dl)] = {
                'n_significant': n_sig,
                'n_cells_measured': int(acc.n),
                'top_effects': top_effects,
            }
            attenuation_curve.append(n_sig)

        max_sig = max(attenuation_curve) if attenuation_curve else 1
        attenuation_curve_norm = [v / max_sig for v in attenuation_curve] if max_sig > 0 else attenuation_curve

        result = {
            'source_feature_idx': fi,
            'source_label': sf['label'],
            'source_n_ontologies': sf['n_ontologies'],
            'source_n_annotations': sf['n_annotations'],
            'source_activation_freq': sf['activation_freq'],
            'n_cells_with_activity': n_cells_active[fi],
            'downstream_effects': downstream_effects,
            'attenuation_curve_raw': attenuation_curve,
            'attenuation_curve_normalized': attenuation_curve_norm,
            'total_significant_edges': total_sig_edges,
        }
        feature_results.append(result)

        print(f"  Feature {fi} ({sf['label'][:40]}): "
              f"active in {n_cells_active[fi]} cells, "
              f"{total_sig_edges} sig edges")

    # Final save
    total_elapsed = time.time() - total_t0
    output = {
        'source_layer': source_layer,
        'config': {
            'n_source_features': len(selected_features),
            'n_cells': n_cells,
            'downstream_layers': downstream_layers,
            'cohens_d_threshold': COHENS_D_THRESHOLD,
            'consistency_threshold': CONSISTENCY_THRESHOLD,
        },
        'stats': {
            'n_clean_passes': n_clean_passes,
            'n_ablated_passes': n_ablated_passes,
            'total_forward_passes': n_clean_passes + n_ablated_passes,
        },
        'features': feature_results,
        'total_compute_time_sec': total_elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(final_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    print(f"  Source layer {source_layer} complete: {total_elapsed/60:.1f} min")
    print(f"  Forward passes: {n_clean_passes} clean + {n_ablated_passes} ablated = {n_clean_passes + n_ablated_passes}")
    print(f"  Saved: {final_path}")

    return final_path


def _save_partial_v2(path, source_layer, cells_completed, feature_indices,
                     accumulators, n_cells_active, n_cells):
    """Save incremental accumulator state for resume capability."""
    acc_state = []
    for fi in feature_indices:
        fi_accs = {}
        for dl, acc in accumulators[fi].items():
            fi_accs[str(dl)] = {
                'n': int(acc.n),
                'mean': acc.mean.tolist(),
                'M2': acc.M2.tolist(),
                'pos_count': acc.pos_count.tolist(),
            }
        acc_state.append({
            'feature_idx': fi,
            'n_cells_active': n_cells_active[fi],
            'accumulators': fi_accs,
        })

    output = {
        'source_layer': source_layer,
        'cells_completed': cells_completed,
        'n_cells': n_cells,
        'accumulator_state': acc_state,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"    Partial save at cell {cells_completed}: {path}")


# ============================================================
# Post-compute analysis
# ============================================================

def build_circuit_graph(out_dir, source_layers):
    """Aggregate significant edges into a circuit graph."""
    print("\n  Building circuit graph...")

    all_edges = []
    feature_labels = {}  # (layer, idx) → label

    for sl in source_layers:
        path = os.path.join(out_dir, f"circuit_L{sl:02d}_features.json")
        if not os.path.exists(path):
            print(f"    Skipping L{sl} — not found")
            continue

        with open(path) as f:
            data = json.load(f)

        for feat in data['features']:
            src_key = f"L{sl}_F{feat['source_feature_idx']}"
            feature_labels[(sl, feat['source_feature_idx'])] = feat['source_label']

            for dl_str, effects in feat['downstream_effects'].items():
                dl = int(dl_str)
                for eff in effects.get('top_effects', []):
                    all_edges.append({
                        'source_layer': sl,
                        'source_feature': feat['source_feature_idx'],
                        'source_label': feat['source_label'],
                        'target_layer': dl,
                        'target_feature': eff['target_feature_idx'],
                        'cohens_d': eff['cohens_d'],
                        'consistency': eff['consistency'],
                        'mean_delta': eff['mean_delta'],
                    })

    print(f"    Total significant edges: {len(all_edges)}")

    # Compute graph statistics
    in_degree = {}
    out_degree = {}
    for e in all_edges:
        src = f"L{e['source_layer']}_F{e['source_feature']}"
        tgt = f"L{e['target_layer']}_F{e['target_feature']}"
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # Top hub features
    top_out = sorted(out_degree.items(), key=lambda x: -x[1])[:20]
    top_in = sorted(in_degree.items(), key=lambda x: -x[1])[:20]

    graph = {
        'n_edges': len(all_edges),
        'n_source_features': len(set(f"L{e['source_layer']}_F{e['source_feature']}" for e in all_edges)),
        'n_target_features': len(set(f"L{e['target_layer']}_F{e['target_feature']}" for e in all_edges)),
        'top_hub_sources': [{'feature': k, 'out_degree': v} for k, v in top_out],
        'top_hub_targets': [{'feature': k, 'in_degree': v} for k, v in top_in],
        'edges': all_edges,
    }

    out_path = os.path.join(out_dir, "circuit_graph.json")
    with open(out_path, 'w') as f:
        json.dump(graph, f, indent=2, default=_json_default)
    print(f"    Saved: {out_path}")

    return graph


def compare_with_pmi(out_dir, source_layers):
    """Compare causal edges with PMI-based statistical dependencies."""
    print("\n  Comparing with PMI...")
    comp_graph_dir = os.path.join(DATA_DIR, "computational_graph")

    # PMI layer pairs available
    pmi_pairs = {
        (0, 5): "deps_L00_to_L05.json",
        (5, 11): "deps_L05_to_L11.json",
        (11, 17): "deps_L11_to_L17.json",
    }

    comparisons = {}

    for (pmi_src, pmi_dst), pmi_file in pmi_pairs.items():
        pmi_path = os.path.join(comp_graph_dir, pmi_file)
        if not os.path.exists(pmi_path):
            continue

        with open(pmi_path) as f:
            pmi_data = json.load(f)

        # Build PMI edge set — format: {dependencies: [{feature_a, top_dependencies: [{feature_b, pmi}]}]}
        pmi_edges = set()
        deps = pmi_data.get('dependencies', [])
        for dep in deps:
            fa = dep.get('feature_a')
            if fa is None:
                continue
            for td in dep.get('top_dependencies', []):
                fb = td.get('feature_b')
                if fb is not None:
                    pmi_edges.add((int(fa), int(fb)))

        # Load causal edges for this layer pair
        causal_path = os.path.join(out_dir, f"circuit_L{pmi_src:02d}_features.json")
        if not os.path.exists(causal_path):
            continue

        with open(causal_path) as f:
            causal_data = json.load(f)

        causal_edges = set()
        for feat in causal_data['features']:
            dl_str = str(pmi_dst)
            if dl_str in feat['downstream_effects']:
                for eff in feat['downstream_effects'][dl_str].get('top_effects', []):
                    causal_edges.add((feat['source_feature_idx'], eff['target_feature_idx']))

        overlap = pmi_edges & causal_edges
        causal_only = causal_edges - pmi_edges
        pmi_only = pmi_edges - causal_edges

        comp = {
            'layer_pair': f"L{pmi_src}→L{pmi_dst}",
            'n_pmi_edges': len(pmi_edges),
            'n_causal_edges': len(causal_edges),
            'n_overlap': len(overlap),
            'n_causal_only': len(causal_only),
            'n_pmi_only': len(pmi_only),
            'overlap_frac_of_causal': len(overlap) / max(len(causal_edges), 1),
            'overlap_frac_of_pmi': len(overlap) / max(len(pmi_edges), 1),
        }
        comparisons[f"L{pmi_src}_to_L{pmi_dst}"] = comp
        print(f"    L{pmi_src}→L{pmi_dst}: "
              f"PMI={len(pmi_edges)}, Causal={len(causal_edges)}, "
              f"Overlap={len(overlap)} ({comp['overlap_frac_of_causal']:.1%} of causal)")

    return comparisons


def analyze_biology(out_dir, source_layers):
    """Look for biological motifs in the circuit graph."""
    print("\n  Analyzing biological motifs...")

    # Load annotations for all layers
    layer_annotations = {}
    for l in range(N_LAYERS):
        ann_path = os.path.join(SAE_BASE, f"layer{l:02d}_x{EXPANSION}_k{K_VAL}", "feature_annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                ann_data = json.load(f)
            layer_annotations[l] = ann_data.get('feature_annotations', {})

    # Load circuit graph
    graph_path = os.path.join(out_dir, "circuit_graph.json")
    with open(graph_path) as f:
        graph = json.load(f)

    # For each edge, look up annotations of both source and target
    annotated_edges = []
    shared_ontology_count = 0
    total_annotated_pairs = 0

    for edge in graph['edges']:
        sl, sf = edge['source_layer'], edge['source_feature']
        tl, tf = edge['target_layer'], edge['target_feature']

        src_anns = layer_annotations.get(sl, {}).get(str(sf), [])
        tgt_anns = layer_annotations.get(tl, {}).get(str(tf), [])

        if src_anns and tgt_anns:
            total_annotated_pairs += 1
            src_terms = set(a['term'] for a in src_anns)
            tgt_terms = set(a['term'] for a in tgt_anns)
            shared = src_terms & tgt_terms

            if shared:
                shared_ontology_count += 1

            # Get best labels
            src_label = edge['source_label']
            tgt_label = "unknown"
            for a in tgt_anns:
                if a['ontology'] in ('GO_BP', 'KEGG', 'Reactome'):
                    tgt_label = a['term']
                    break

            annotated_edges.append({
                'source': f"L{sl}_F{sf}",
                'target': f"L{tl}_F{tf}",
                'source_label': src_label,
                'target_label': tgt_label,
                'shared_terms': list(shared) if shared else [],
                'cohens_d': edge['cohens_d'],
                'n_source_annotations': len(src_anns),
                'n_target_annotations': len(tgt_anns),
            })

    # Sort by number of shared terms and Cohen's d
    annotated_edges.sort(key=lambda x: (-len(x['shared_terms']), -abs(x['cohens_d'])))

    bio_motifs = {
        'total_edges': len(graph['edges']),
        'total_annotated_pairs': total_annotated_pairs,
        'shared_ontology_edges': shared_ontology_count,
        'shared_ontology_frac': shared_ontology_count / max(total_annotated_pairs, 1),
        'top_interpretable_circuits': annotated_edges[:50],
    }

    print(f"    Annotated edge pairs: {total_annotated_pairs}")
    print(f"    Shared ontology terms: {shared_ontology_count} ({bio_motifs['shared_ontology_frac']:.1%})")

    if annotated_edges:
        print(f"    Top circuit examples:")
        for ae in annotated_edges[:5]:
            shared_str = ", ".join(ae['shared_terms'][:3]) if ae['shared_terms'] else "none"
            print(f"      {ae['source']} ({ae['source_label'][:30]}) → "
                  f"{ae['target']} ({ae['target_label'][:30]}) | "
                  f"d={ae['cohens_d']:.2f} | shared: {shared_str}")

    return bio_motifs


def analyze_attenuation(out_dir, source_layers):
    """Analyze how effects attenuate across layers."""
    print("\n  Analyzing attenuation curves...")

    attenuation_data = {}

    for sl in source_layers:
        path = os.path.join(out_dir, f"circuit_L{sl:02d}_features.json")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            data = json.load(f)

        layer_curves = []
        for feat in data['features']:
            if feat['total_significant_edges'] > 0:
                layer_curves.append({
                    'feature_idx': feat['source_feature_idx'],
                    'label': feat['source_label'],
                    'total_edges': feat['total_significant_edges'],
                    'curve_raw': feat['attenuation_curve_raw'],
                    'curve_normalized': feat['attenuation_curve_normalized'],
                })

        # Compute average attenuation curve for this source layer
        if layer_curves:
            max_len = max(len(c['curve_raw']) for c in layer_curves)
            avg_curve = np.zeros(max_len)
            count = np.zeros(max_len)
            for c in layer_curves:
                for i, v in enumerate(c['curve_raw']):
                    avg_curve[i] += v
                    count[i] += 1
            avg_curve = avg_curve / np.maximum(count, 1)

            attenuation_data[f"L{sl}"] = {
                'n_features_with_effects': len(layer_curves),
                'avg_attenuation_curve': avg_curve.tolist(),
                'features': layer_curves,
            }

            print(f"    L{sl}: {len(layer_curves)} features with effects, "
                  f"avg sig edges per downstream layer: {avg_curve.mean():.1f}")

    return attenuation_data


def run_analysis(out_dir, source_layers):
    """Run all post-compute analyses."""
    print("\n" + "=" * 70)
    print("POST-COMPUTE ANALYSIS")
    print("=" * 70)

    graph = build_circuit_graph(out_dir, source_layers)
    pmi_comparison = compare_with_pmi(out_dir, source_layers)
    bio_motifs = analyze_biology(out_dir, source_layers)
    attenuation = analyze_attenuation(out_dir, source_layers)

    analysis = {
        'source_layers': source_layers,
        'circuit_graph_summary': {
            'n_edges': graph['n_edges'],
            'n_source_features': graph['n_source_features'],
            'n_target_features': graph['n_target_features'],
        },
        'pmi_comparison': pmi_comparison,
        'biological_motifs': bio_motifs,
        'attenuation': attenuation,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = os.path.join(out_dir, "circuit_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=_json_default)
    print(f"\n  Analysis saved: {out_path}")

    return analysis


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Causal feature-to-feature circuit tracing")
    parser.add_argument('--source-layers', type=str, default='0,5,11,15',
                        help='Comma-separated source layers (default: 0,5,11,15)')
    parser.add_argument('--n-features', type=int, default=30,
                        help='Number of source features per layer (default: 30)')
    parser.add_argument('--n-cells', type=int, default=200,
                        help='Number of cells to trace through (default: 200)')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Skip computation, only run analysis on existing results')
    parser.add_argument('--sae-dir', type=str, default=None,
                        help='SAE models directory (default: experiments/phase1_k562/sae_models)')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: experiments/phase1_k562/circuit_tracing)')
    parser.add_argument('--available-layers', type=str, default=None,
                        help='Comma-separated layers with SAE models (default: all 0-17). '
                             'Downstream tracing is restricted to these layers.')
    parser.add_argument('--data-source', type=str, default='k562',
                        choices=['k562', 'tabula_sapiens'],
                        help='Cell data source: k562 (Replogle CRISPRi controls) or '
                             'tabula_sapiens (immune+kidney+lung)')
    args = parser.parse_args()

    global SAE_BASE
    if args.sae_dir:
        SAE_BASE = args.sae_dir

    source_layers = [int(x) for x in args.source_layers.split(',')]
    available_layers = None
    if args.available_layers:
        available_layers = [int(x) for x in args.available_layers.split(',')]

    out_dir = args.out_dir or os.path.join(DATA_DIR, "circuit_tracing")
    os.makedirs(out_dir, exist_ok=True)

    total_t0 = time.time()

    print("=" * 70)
    print("STEP 13: CAUSAL FEATURE-TO-FEATURE CIRCUIT TRACING")
    print(f"  Source layers: {source_layers}")
    print(f"  Features per layer: {args.n_features}")
    print(f"  Cells: {args.n_cells}")
    print(f"  Data source: {args.data_source}")
    print(f"  SAE dir: {SAE_BASE}")
    if available_layers:
        print(f"  Available SAE layers: {available_layers}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    if not args.analysis_only:
        # Load model and tokenize cells (shared across all source layers)
        import torch
        from transformers import BertForMaskedLM

        print("\n  Loading Geneformer V2-316M...")
        t0 = time.time()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("    Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("    Using CPU")

        model = BertForMaskedLM.from_pretrained(
            MODEL_NAME, subfolder=MODEL_SUBFOLDER,
            output_hidden_states=True,
            output_attentions=False,
            attn_implementation="eager",
        )
        model = model.to(device)
        model.eval()
        print(f"    Model loaded in {time.time()-t0:.1f}s")

        # Tokenize cells
        all_tokens = load_and_tokenize_cells(args.n_cells, data_source=args.data_source)

        # SAE cache
        sae_cache = SAECache()

        # Process each source layer
        for sl in source_layers:
            print(f"\n{'=' * 70}")
            print(f"SOURCE LAYER {sl}")
            print(f"{'=' * 70}")

            selected = select_features(sl, args.n_features)
            trace_source_layer(sl, selected, all_tokens, model, device,
                               sae_cache, out_dir, args.n_cells,
                               available_layers=available_layers)

            # Clear GPU cache between source layers
            if device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()

    # Run analysis
    run_analysis(out_dir, source_layers)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hrs)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
