#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import anndata as ad
import numpy as np
from torchdiffeq import odeint
from sklearn.neighbors import NearestNeighbors

# 导入模型定义
from myUtility import UOT


def propagate_cells_ode(func, z_start, t_start, t_end, device):
    if abs(t_start - t_end) < 1e-8:
        return z_start
    z_start = z_start.clone().detach().requires_grad_(False)
    batch_size = z_start.shape[0]
    g_start = torch.zeros(batch_size, 1, device=device)
    logp_start = torch.zeros(batch_size, 1, device=device)
    t_span = torch.tensor([t_start, t_end], dtype=torch.float32, device=device)

    def odefunc(t, state):
        dz_dt, g_out, dlogp_dt = func(t, state)
        return (dz_dt, torch.zeros_like(g_out), torch.zeros_like(dlogp_dt))

    with torch.no_grad():
        sol = odeint(odefunc, (z_start, g_start, logp_start), t_span, method='midpoint', atol=1e-5, rtol=1e-5)
        return sol[0][-1]


def add_trajectory_links_to_adata(adata, func, embed_key, time_key, device):
    if embed_key not in adata.obsm:
        raise KeyError(f"Embedding '{embed_key}' not in adata.obsm")
    if time_key not in adata.obs:
        raise KeyError(f"Time column '{time_key}' not in adata.obs")

    X_embed = np.array(adata.obsm[embed_key])
    times = adata.obs[time_key].values
    X_tensor = torch.from_numpy(X_embed).float().to(device)

    unique_times = np.sort(np.unique(times))
    if len(unique_times) < 2:
        raise ValueError("Need at least two time points")

    N_total = len(times)
    prev_id = np.full(N_total, -1, dtype=int)
    next_id = np.full(N_total, -1, dtype=int)

    time_to_indices = {t: np.where(times == t)[0] for t in unique_times}

    for i, t_current in enumerate(unique_times):
        idx_current = time_to_indices[t_current]
        cells_current = X_tensor[idx_current]

        # Next
        if i < len(unique_times) - 1:
            t_next = unique_times[i + 1]
            idx_next = time_to_indices[t_next]
            cells_next_real = X_tensor[idx_next]
            cells_prop = propagate_cells_ode(func, cells_current, t_current, t_next, device)
            nbrs = NearestNeighbors(n_neighbors=1).fit(cells_next_real.cpu().numpy())
            _, indices = nbrs.kneighbors(cells_prop.cpu().numpy())
            next_id[idx_current] = idx_next[indices.flatten()]

        # Prev
        if i > 0:
            t_prev = unique_times[i - 1]
            idx_prev = time_to_indices[t_prev]
            cells_prev_real = X_tensor[idx_prev]
            cells_prop_back = propagate_cells_ode(func, cells_current, t_current, t_prev, device)
            nbrs = NearestNeighbors(n_neighbors=1).fit(cells_prev_real.cpu().numpy())
            _, indices = nbrs.kneighbors(cells_prop_back.cpu().numpy())
            prev_id[idx_current] = idx_prev[indices.flatten()]

    adata.obs['prev_cell_id'] = prev_id
    adata.obs['next_cell_id'] = next_id
    return adata


def main():
    parser = argparse.ArgumentParser(description="Add trajectory links to h5ad using trained UOT model.")
    parser.add_argument("--input_h5ad", type=str, required=True, help="Input .h5ad file path")
    parser.add_argument("--output_h5ad", type=str, required=True, help="Output .h5ad file path")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--embed_key", type=str, default="X_AE", help="Key in adata.obsm for embedding (default: X_AE)")
    parser.add_argument("--time_key", type=str, default="time", help="Key in adata.obs for time (default: time)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (e.g., cuda:0 or cpu)")
    parser.add_argument("--in_out_dim", type=int, default=10, help="Dimension of embedding space (default: 10)")
    parser.add_argument("--hidden_dim", type=int, default=16, help="Hidden dimension (default: 16)")
    parser.add_argument("--n_hiddens", type=int, default=4, help="Number of hidden layers (default: 4)")
    parser.add_argument("--activation", type=str, default="Tanh", help="Activation function (default: Tanh)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    print(f"Using device: {device}")

    # Load model with EXACT same architecture as training
    print("Building model with:")
    print(f"  in_out_dim={args.in_out_dim}, hidden_dim={args.hidden_dim}, n_hiddens={args.n_hiddens}, activation={args.activation}")
    model = UOT(
        in_out_dim=args.in_out_dim,
        hidden_dim=args.hidden_dim,
        n_hiddens=args.n_hiddens,
        activation=args.activation
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if 'func_state_dict' not in ckpt:
        raise KeyError(f"'func_state_dict' not found in ckpt. Keys: {list(ckpt.keys())}")
    
    model.load_state_dict(ckpt['func_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")


    # Load data
    print(f"Loading data from {args.input_h5ad}...")
    adata = ad.read_h5ad(args.input_h5ad)

    # Add links
    print("Adding trajectory links...")
    adata = add_trajectory_links_to_adata(
        adata=adata,
        func=model,
        embed_key=args.embed_key,
        time_key=args.time_key,
        device=device
    )

    # Save
    print(f"Saving to {args.output_h5ad}...")
    adata.write_h5ad(args.output_h5ad)
    print("Done!")


if __name__ == "__main__":
    main()
