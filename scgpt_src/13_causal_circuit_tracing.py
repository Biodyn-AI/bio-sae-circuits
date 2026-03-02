#!/usr/bin/env python3
"""
Phase 4: Causal Feature-to-Feature Circuit Tracing (scGPT).

Ablate individual SAE features at source layers and measure how downstream
SAE features change — revealing directed information flow through the model.

For each source feature:
  1. Run clean forward pass → capture hidden states at ALL downstream layers
  2. Encode downstream hidden states through downstream SAEs → clean activations
  3. Ablate source feature: SAE encode → zero feature → decode → compute delta
  4. Run ablated forward through remaining layers → capture downstream states
  5. Encode ablated hidden states through downstream SAEs → ablated activations
  6. Accumulate per-downstream-feature deltas across cells (Welford's algorithm)
  7. Compute significance: Cohen's d + consistency

Prerequisites (from bio-sae, Phases 1-3):
  - scGPT trained SAE checkpoints: experiments/scgpt_atlas/sae_models/layer{NN}_x4_k32/sae_final.pt
  - scGPT model checkpoint (whole-human, best_model.pt + vocab.json)
  - Tabula Sapiens cell data (h5ad)

Configuration:
  Set BASE below to your local root directory.
  Adjust PROJ_DIR, SAE_BASE, SCGPT_CHECKPOINT, SCGPT_VOCAB for your setup.

Usage:
    python scgpt_src/13_causal_circuit_tracing.py \
        [--source-layers 0,4,8] [--n-features 30] [--n-cells 200]
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np

BASE = "/Volumes/Crucial X6/MacBook/biomechinterp"
PROJ_DIR = os.path.join(BASE, "biodyn-work/subproject_42_sparse_autoencoder_biological_map")
MECHINTERP_DIR = os.path.join(BASE, "biodyn-work/single_cell_mechinterp")
SAE_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/sae_models")
OUT_BASE = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/circuit_tracing")

# scGPT model
SCGPT_REPO = os.path.join(MECHINTERP_DIR, "external/scGPT")
SCGPT_CHECKPOINT = os.path.join(MECHINTERP_DIR, "external/scGPT_checkpoints/whole-human/best_model.pt")
SCGPT_VOCAB = os.path.join(MECHINTERP_DIR, "external/scGPT_checkpoints/whole-human/vocab.json")

# Model architecture
D_MODEL = 512
N_LAYERS = 12
N_HEADS = 8
D_HID = 512
DROPOUT = 0.2
MAX_SEQ_LEN = 1200

# SAE config
EXPANSION = 4
K_VAL = 32
N_FEATURES = D_MODEL * EXPANSION  # 2048

# Significance thresholds
COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7
TOP_EFFECTS_PER_LAYER = 50

# Cell data (Tabula Sapiens)
TISSUES = {
    'immune': os.path.join(MECHINTERP_DIR, "data/raw/tabula_sapiens_immune_subset_20000.h5ad"),
    'kidney': os.path.join(MECHINTERP_DIR, "data/raw/tabula_sapiens_kidney.h5ad"),
    'lung': os.path.join(MECHINTERP_DIR, "data/raw/tabula_sapiens_lung.h5ad"),
}
EXTRACTION_META = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/activations/extraction_metadata.json")


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


def build_gene_name_map(h5_path):
    """Build gene index -> gene symbol mapping from h5ad file."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        if 'feature_name' in f['var']:
            fn = f['var']['feature_name']
            if isinstance(fn, h5py.Group):
                categories = fn['categories'][:]
                codes = fn['codes'][:]
                if categories.dtype.kind in ('O', 'S'):
                    categories = np.array([x.decode() if isinstance(x, bytes) else x for x in categories])
                gene_names = categories[codes]
            else:
                gene_names = fn[:]
                if gene_names.dtype.kind in ('O', 'S'):
                    gene_names = np.array([x.decode() if isinstance(x, bytes) else x for x in gene_names])
        else:
            var_index = f['var']['_index'][:]
            gene_names = np.array([x.decode() if isinstance(x, bytes) else x for x in var_index])
        n_genes = len(gene_names)
    return gene_names, n_genes


def tokenize_cell_scgpt(expression_vector, gene_names, vocab, pad_token_id,
                        max_seq_len=1200, pad_value=-2):
    """Tokenize a single cell for scGPT."""
    nonzero_mask = expression_vector > 0
    nonzero_indices = np.where(nonzero_mask)[0]

    if len(nonzero_indices) == 0:
        return None

    valid_token_ids = []
    valid_expr = []
    valid_names = []

    for idx in nonzero_indices:
        gname = gene_names[idx]
        if gname in vocab:
            valid_token_ids.append(vocab[gname])
            valid_expr.append(expression_vector[idx])
            valid_names.append(gname)

    if len(valid_token_ids) == 0:
        return None

    valid_token_ids = np.array(valid_token_ids, dtype=np.int64)
    valid_expr = np.array(valid_expr, dtype=np.float32)

    # Sort by expression descending
    order = np.argsort(-valid_expr)
    valid_token_ids = valid_token_ids[order]
    valid_expr = valid_expr[order]
    valid_names = [valid_names[i] for i in order]

    if len(valid_token_ids) > max_seq_len:
        valid_token_ids = valid_token_ids[:max_seq_len]
        valid_expr = valid_expr[:max_seq_len]
        valid_names = valid_names[:max_seq_len]

    n_genes = len(valid_token_ids)
    pad_len = max_seq_len - n_genes
    gene_ids = np.pad(valid_token_ids, (0, pad_len), mode='constant',
                      constant_values=pad_token_id)
    gene_values = np.pad(valid_expr, (0, pad_len), mode='constant',
                         constant_values=pad_value)
    src_key_padding_mask = np.zeros(max_seq_len, dtype=bool)
    src_key_padding_mask[n_genes:] = True

    return {
        'gene_ids': gene_ids,
        'gene_values': gene_values,
        'src_key_padding_mask': src_key_padding_mask,
        'n_genes': n_genes,
        'gene_names': valid_names,
    }


def select_features(layer, n_features=30):
    """Select well-annotated features for circuit tracing."""
    run_name = f"layer{layer:02d}_x{EXPANSION}_k{K_VAL}"
    run_dir = os.path.join(SAE_BASE, run_name)

    with open(os.path.join(run_dir, "feature_annotations.json")) as f:
        ann_data = json.load(f)
    with open(os.path.join(run_dir, "feature_catalog.json")) as f:
        catalog = json.load(f)

    feature_annotations = ann_data.get('feature_annotations', {})

    feature_genes = {}
    feature_freq = {}
    for feat in catalog['features']:
        fi = feat['feature_idx']
        if feat.get('top_genes'):
            feature_genes[fi] = [g['gene_name'] for g in feat['top_genes'][:20]]
        feature_freq[fi] = feat.get('activation_freq', 0)

    scored = []
    for fid_str, anns in feature_annotations.items():
        fi = int(fid_str)
        if fi not in feature_genes or len(feature_genes[fi]) < 5:
            continue
        if feature_freq.get(fi, 0) < 0.01:
            continue

        ontologies = set(a['ontology'] for a in anns if 'ontology' in a)
        n_ont = len(ontologies)
        n_ann = len([a for a in anns if 'p_adjusted' in a])
        n_edge = len([a for a in anns if 'n_edges' in a])
        min_p = min((a.get('p_adjusted', 1.0) for a in anns), default=1.0)

        best_label = "unknown"
        for a in anns:
            if a.get('ontology') in ('GO_BP', 'KEGG', 'Reactome'):
                best_label = a['term']
                break

        score = n_ont * 10 + n_ann + n_edge * 2 - np.log10(max(min_p, 1e-30))

        scored.append({
            'feature_idx': fi,
            'n_ontologies': n_ont,
            'n_annotations': n_ann + n_edge,
            'min_p': min_p,
            'label': best_label,
            'top_genes': feature_genes[fi],
            'activation_freq': feature_freq.get(fi, 0),
            'score': score,
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
        self.n += 1
        d = delta_vector - self.mean
        self.mean += d / self.n
        d2 = delta_vector - self.mean
        self.M2 += d * d2
        self.pos_count += (delta_vector > 0).astype(np.float64)

    def finalize(self):
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
        if layer not in self._saes:
            import torch
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
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
        for l in layers:
            self.get(l)


def load_and_tokenize_cells(n_cells, vocab, pad_token_id):
    """Load and tokenize Tabula Sapiens cells for scGPT."""
    import h5py

    print("  Loading Tabula Sapiens cells for scGPT...")

    # Load extraction metadata to get the same cell selection
    with open(EXTRACTION_META) as f:
        meta = json.load(f)
    cell_data = meta['cell_data']  # List of {tissue, cell_type, cell_idx}

    # Subsample with stratified tissue sampling
    rng = np.random.RandomState(42)
    per_tissue = n_cells // 3
    remainder = n_cells - per_tissue * 3

    tissue_cells = {'immune': [], 'kidney': [], 'lung': []}
    for cd in cell_data:
        tissue_cells[cd['tissue']].append(cd)

    selected_cells = []
    for ti, (tissue_name, cells) in enumerate(tissue_cells.items()):
        n_this = per_tissue + (1 if ti < remainder else 0)
        if len(cells) > n_this:
            chosen = rng.choice(len(cells), n_this, replace=False)
            selected_cells.extend([cells[i] for i in sorted(chosen)])
        else:
            selected_cells.extend(cells[:n_this])

    print(f"    Selected {len(selected_cells)} cells from {len(tissue_cells)} tissues")

    # Build gene name maps per tissue
    tissue_gene_data = {}
    for tissue_name, h5_path in TISSUES.items():
        gene_names, n_genes = build_gene_name_map(h5_path)
        tissue_gene_data[tissue_name] = {'gene_names': gene_names, 'n_genes': n_genes}

    # Tokenize each cell
    all_tokenized = []
    tissue_counts = {}
    for cd in selected_cells:
        tissue = cd['tissue']
        cell_idx = cd['cell_idx']
        gdata = tissue_gene_data[tissue]

        with h5py.File(TISSUES[tissue], 'r') as f:
            expr = load_sparse_row(f['X'], cell_idx, gdata['n_genes'])

        result = tokenize_cell_scgpt(
            expr, gdata['gene_names'], vocab, pad_token_id,
            max_seq_len=MAX_SEQ_LEN, pad_value=-2
        )

        if result is not None:
            all_tokenized.append(result)
            tissue_counts[tissue] = tissue_counts.get(tissue, 0) + 1

    for tissue, count in tissue_counts.items():
        print(f"      {tissue}: {count} cells")
    print(f"    Total tokenized: {len(all_tokenized)} cells")

    return all_tokenized


def trace_source_layer(source_layer, selected_features, all_cells, model, device,
                       sae_cache, out_dir, n_cells):
    """Run circuit tracing for all source features at one source layer.

    Iterates over cells in the outer loop. For each cell, runs ONE
    clean forward pass, then processes all features that are active.
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

    # Downstream = all layers after source
    downstream_layers = list(range(source_layer + 1, N_LAYERS))
    print(f"  Source layer {source_layer} → tracing through {len(downstream_layers)} downstream layers: {downstream_layers}")

    # Preload downstream SAEs
    sae_cache.preload(downstream_layers)
    dst_act_mean_tensors = {}
    for dl in downstream_layers:
        _, dst_mean = sae_cache.get(dl)
        dst_act_mean_tensors[dl] = torch.tensor(dst_mean, dtype=torch.float32)

    # Build feature index
    feature_indices = [sf['feature_idx'] for sf in selected_features]
    fi_to_idx = {fi: idx for idx, fi in enumerate(feature_indices)}

    # Initialize Welford accumulators
    accumulators = {}
    n_cells_active = {}
    for sf in selected_features:
        fi = sf['feature_idx']
        accumulators[fi] = {dl: WelfordAccumulator(N_FEATURES) for dl in downstream_layers}
        n_cells_active[fi] = 0

    # Check for partial results (resume)
    start_cell = 0
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        start_cell = partial.get('cells_completed', 0)
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

    cells_to_process = all_cells[:n_cells]
    print(f"  Processing {len(cells_to_process)} cells (starting from {start_cell})...")

    for ci in range(start_cell, len(cells_to_process)):
        cell = cells_to_process[ci]
        n_genes = cell['n_genes']

        if n_genes < 10:
            continue

        gene_ids_t = torch.tensor(cell['gene_ids'], dtype=torch.long).unsqueeze(0).to(device)
        gene_values_t = torch.tensor(cell['gene_values'], dtype=torch.float32).unsqueeze(0).to(device)
        padding_mask = torch.tensor(cell['src_key_padding_mask'], dtype=torch.bool).unsqueeze(0).to(device)

        # === ONE clean forward pass per cell ===
        # Register hooks to capture all layer outputs
        clean_hidden_states = {}

        def make_capture_hook(layer_idx):
            def hook_fn(module, input, output):
                clean_hidden_states[layer_idx] = output.detach().clone()
            return hook_fn

        hooks = []
        for i, layer_mod in enumerate(model.transformer_encoder.layers):
            hooks.append(layer_mod.register_forward_hook(make_capture_hook(i)))

        with torch.no_grad():
            model._encode(src=gene_ids_t, values=gene_values_t,
                          src_key_padding_mask=padding_mask)

        for h in hooks:
            h.remove()
        n_clean_passes += 1

        # Gene positions = non-padded positions
        gene_positions = list(range(n_genes))

        # Get source layer hidden states at gene positions
        clean_hidden_full = clean_hidden_states[source_layer][0].cpu()  # (seq_len, 512)
        gene_hidden = clean_hidden_full[:n_genes]  # (n_genes, 512)
        gene_hidden_centered = gene_hidden - source_act_mean_t

        # Encode through source SAE
        with torch.no_grad():
            h_sparse, topk_indices = source_sae.encode(gene_hidden_centered)

        # Pre-extract clean downstream hidden states at gene positions
        clean_downstream = {}
        for dl in downstream_layers:
            clean_downstream[dl] = clean_hidden_states[dl][0][:n_genes].cpu()

        # Check which features are active and process them
        for fi in feature_indices:
            active_mask = (topk_indices == fi).any(dim=1)
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

            # Build modified hidden state (full sequence length)
            modified_hidden = clean_hidden_full.clone()
            modified_hidden[:n_genes] = clean_hidden_full[:n_genes] + delta

            # --- Ablated forward pass through remaining layers ---
            x = modified_hidden.unsqueeze(0).to(device)
            ablated_downstream = {}

            with torch.no_grad():
                for dl in range(source_layer + 1, N_LAYERS):
                    x = model.transformer_encoder.layers[dl](
                        x, src_key_padding_mask=padding_mask)
                    if dl in downstream_layers:
                        ablated_downstream[dl] = x[0, :n_genes].cpu()

            n_ablated_passes += 1

            # --- Encode downstream layers and accumulate ---
            active_positions = torch.where(active_mask)[0]

            for dl in downstream_layers:
                dst_sae, _ = sae_cache.get(dl)
                dst_mean_t = dst_act_mean_tensors[dl]

                clean_dl = clean_downstream[dl]
                ablated_dl = ablated_downstream[dl]

                with torch.no_grad():
                    clean_sparse, _ = dst_sae.encode(clean_dl - dst_mean_t)
                    ablated_sparse, _ = dst_sae.encode(ablated_dl - dst_mean_t)

                # Average delta across active positions
                pos_deltas = []
                for pos in active_positions:
                    pos_deltas.append(
                        (ablated_sparse[pos] - clean_sparse[pos]).numpy().astype(np.float64)
                    )
                cell_delta = np.mean(pos_deltas, axis=0)
                accumulators[fi][dl].update(cell_delta)

            del ablated_downstream, modified_hidden, h_ablated, delta

        # Cleanup
        del clean_hidden_states, clean_hidden_full, gene_hidden, h_sparse, topk_indices
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
            _save_partial(partial_path, source_layer, ci + 1, feature_indices,
                          accumulators, n_cells_active, n_cells)

    # === Finalize: compute significance ===
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
        attenuation_norm = [v / max_sig for v in attenuation_curve] if max_sig > 0 else attenuation_curve

        result = {
            'source_feature_idx': fi,
            'source_label': sf['label'],
            'source_n_ontologies': sf['n_ontologies'],
            'source_n_annotations': sf['n_annotations'],
            'source_activation_freq': sf['activation_freq'],
            'n_cells_with_activity': n_cells_active[fi],
            'downstream_effects': downstream_effects,
            'attenuation_curve_raw': attenuation_curve,
            'attenuation_curve_normalized': attenuation_norm,
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
        'model': 'scGPT whole-human',
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


def _save_partial(path, source_layer, cells_completed, feature_indices,
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
    feature_labels = {}

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

    # Graph statistics
    in_degree = {}
    out_degree = {}
    for e in all_edges:
        src = f"L{e['source_layer']}_F{e['source_feature']}"
        tgt = f"L{e['target_layer']}_F{e['target_feature']}"
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

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
    comp_graph_dir = os.path.join(PROJ_DIR, "experiments/scgpt_atlas/computational_graph")

    # scGPT PMI layer pairs
    pmi_pairs = {
        (0, 4): "graph_L00_L04.json",
        (4, 8): "graph_L04_L08.json",
        (8, 11): "graph_L08_L11.json",
    }

    comparisons = {}

    for (pmi_src, pmi_dst), pmi_file in pmi_pairs.items():
        pmi_path = os.path.join(comp_graph_dir, pmi_file)
        if not os.path.exists(pmi_path):
            continue

        with open(pmi_path) as f:
            pmi_data = json.load(f)

        # Build PMI edge set from top_edges
        pmi_edges = set()
        for edge in pmi_data.get('top_edges', []):
            pmi_edges.add((int(edge['upstream']), int(edge['downstream'])))

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

    layer_annotations = {}
    for l in range(N_LAYERS):
        ann_path = os.path.join(SAE_BASE, f"layer{l:02d}_x{EXPANSION}_k{K_VAL}", "feature_annotations.json")
        if os.path.exists(ann_path):
            with open(ann_path) as f:
                ann_data = json.load(f)
            layer_annotations[l] = ann_data.get('feature_annotations', {})

    graph_path = os.path.join(out_dir, "circuit_graph.json")
    with open(graph_path) as f:
        graph = json.load(f)

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
            src_terms = set(a['term'] for a in src_anns if 'term' in a)
            tgt_terms = set(a['term'] for a in tgt_anns if 'term' in a)
            shared = src_terms & tgt_terms

            if shared:
                shared_ontology_count += 1

            src_label = edge['source_label']
            tgt_label = "unknown"
            for a in tgt_anns:
                if a.get('ontology') in ('GO_BP', 'KEGG', 'Reactome'):
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


def analyze_effect_sizes(out_dir, source_layers):
    """Compute aggregate effect size statistics."""
    print("\n  Analyzing effect sizes...")

    all_d_values = []
    n_inhibitory = 0
    n_excitatory = 0

    for sl in source_layers:
        path = os.path.join(out_dir, f"circuit_L{sl:02d}_features.json")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            data = json.load(f)

        for feat in data['features']:
            for dl_str, effects in feat['downstream_effects'].items():
                for eff in effects.get('top_effects', []):
                    d = eff['cohens_d']
                    all_d_values.append(abs(d))
                    if d < 0:
                        n_inhibitory += 1
                    else:
                        n_excitatory += 1

    if not all_d_values:
        return {}

    all_d = np.array(all_d_values)
    total = n_inhibitory + n_excitatory

    stats = {
        'mean_abs_d': float(np.mean(all_d)),
        'median_abs_d': float(np.median(all_d)),
        'max_abs_d': float(np.max(all_d)),
        'n_strong_d_gt_1': int((all_d > 1.0).sum()),
        'pct_strong_d_gt_1': float((all_d > 1.0).mean() * 100),
        'n_very_strong_d_gt_2': int((all_d > 2.0).sum()),
        'pct_very_strong_d_gt_2': float((all_d > 2.0).mean() * 100),
        'n_inhibitory': n_inhibitory,
        'n_excitatory': n_excitatory,
        'inhibitory_pct': float(n_inhibitory / max(total, 1) * 100),
    }

    print(f"    Mean |d|: {stats['mean_abs_d']:.2f}, Median |d|: {stats['median_abs_d']:.2f}")
    print(f"    |d| > 1.0: {stats['pct_strong_d_gt_1']:.1f}%")
    print(f"    Inhibitory: {stats['inhibitory_pct']:.1f}%")

    return stats


def run_analysis(out_dir, source_layers):
    """Run all post-compute analyses."""
    print("\n" + "=" * 70)
    print("POST-COMPUTE ANALYSIS")
    print("=" * 70)

    graph = build_circuit_graph(out_dir, source_layers)
    pmi_comparison = compare_with_pmi(out_dir, source_layers)
    bio_motifs = analyze_biology(out_dir, source_layers)
    attenuation = analyze_attenuation(out_dir, source_layers)
    effect_sizes = analyze_effect_sizes(out_dir, source_layers)

    analysis = {
        'model': 'scGPT whole-human',
        'source_layers': source_layers,
        'circuit_graph_summary': {
            'n_edges': graph['n_edges'],
            'n_source_features': graph['n_source_features'],
            'n_target_features': graph['n_target_features'],
        },
        'effect_sizes': effect_sizes,
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
    parser = argparse.ArgumentParser(description="scGPT causal feature-to-feature circuit tracing")
    parser.add_argument('--source-layers', type=str, default='0,4,8',
                        help='Comma-separated source layers (default: 0,4,8)')
    parser.add_argument('--n-features', type=int, default=30,
                        help='Number of source features per layer (default: 30)')
    parser.add_argument('--n-cells', type=int, default=200,
                        help='Number of cells to trace through (default: 200)')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Skip computation, only run analysis on existing results')
    args = parser.parse_args()

    source_layers = [int(x) for x in args.source_layers.split(',')]
    os.makedirs(OUT_BASE, exist_ok=True)

    total_t0 = time.time()

    print("=" * 70)
    print("scGPT SAE — STEP 13: CAUSAL FEATURE-TO-FEATURE CIRCUIT TRACING")
    print(f"  Model: scGPT whole-human (12L x 8H x 512D)")
    print(f"  Source layers: {source_layers}")
    print(f"  Features per layer: {args.n_features}")
    print(f"  Cells: {args.n_cells}")
    print(f"  SAE dir: {SAE_BASE}")
    print(f"  Output: {OUT_BASE}")
    print("=" * 70)

    if not args.analysis_only:
        import torch

        # Load vocab
        print("\n  Loading scGPT vocab...")
        with open(SCGPT_VOCAB) as f:
            vocab = json.load(f)
        pad_token_id = vocab['<pad>']
        print(f"    Vocab size: {len(vocab)}")

        # Load model
        print("\n  Loading scGPT model...")
        t0 = time.time()

        sys.path.insert(0, SCGPT_REPO)
        from scgpt.model.model import TransformerModel

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("    Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("    Using CPU")

        model = TransformerModel(
            ntoken=len(vocab), d_model=D_MODEL, nhead=N_HEADS,
            d_hid=D_HID, nlayers=N_LAYERS, vocab=vocab,
            dropout=DROPOUT, pad_token="<pad>", pad_value=-2,
            input_emb_style="continuous", use_fast_transformer=False,
            do_mvc=False, do_dab=False, use_batch_labels=False,
            cell_emb_style="avg-pool", n_cls=1)

        checkpoint = torch.load(SCGPT_CHECKPOINT, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict',
                        checkpoint.get('model', checkpoint)))
        converted = {k.replace("Wqkv.", "in_proj_"): v for k, v in state_dict.items()}
        model.load_state_dict(converted, strict=False)
        model = model.to(device)
        model.eval()
        print(f"    Model loaded in {time.time()-t0:.1f}s")

        # Tokenize cells
        all_cells = load_and_tokenize_cells(args.n_cells, vocab, pad_token_id)

        # SAE cache
        sae_cache = SAECache()

        # Process each source layer
        for sl in source_layers:
            print(f"\n{'=' * 70}")
            print(f"SOURCE LAYER {sl}")
            print(f"{'=' * 70}")

            selected = select_features(sl, args.n_features)
            trace_source_layer(sl, selected, all_cells, model, device,
                               sae_cache, OUT_BASE, args.n_cells)

            if device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()

    # Run analysis
    run_analysis(OUT_BASE, source_layers)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hrs)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
