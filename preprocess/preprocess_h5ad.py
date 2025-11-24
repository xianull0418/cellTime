#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import re
import sys
from scipy.stats import median_abs_deviation

# ============ QC 预处理部分 ============

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    mad = median_abs_deviation(M)
    median = np.median(M)
    outlier = (M < median - nmads * mad) | (M > median + nmads * mad)
    return outlier

def preprocess(adata, args):
    print(f"Before QC: {adata.n_obs} cells, {adata.n_vars} genes")

    sc.pp.filter_cells(adata, min_counts=1)  # 确保每个细胞至少有1个UMI
    sc.pp.filter_cells(adata, min_genes=1)   # 可选：至少1个基因

    # QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=[20], log1p=True)

    # Basic cell filtering
    cell_mask = (
        (adata.obs.n_genes_by_counts >= args.min_genes) &
        (adata.obs.n_genes_by_counts <= args.max_genes)
    )
    adata = adata[cell_mask, :].copy()
    sc.pp.filter_genes(adata, min_cells=args.min_cells)

    print(f"Total number of cells after basic filtering: {adata.n_obs}")

    # Outlier removal
    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", 5) |
        is_outlier(adata, "log1p_n_genes_by_counts", 5) |
        is_outlier(adata, "pct_counts_in_top_20_genes", 5)
    )
    adata = adata[~adata.obs.outlier].copy()

    print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")

    # Normalization & log
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection (subset immediately to reduce memory)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_hvg,
        flavor="seurat",
        subset=True,
        inplace=True
    )

    n_hvg_found = adata.var['highly_variable'].sum() if 'highly_variable' in adata.var else adata.n_vars
    print(f"Found {n_hvg_found} highly variable genes.")

    # Scale
    sc.pp.scale(adata)

    print(f"After QC: {adata.n_obs} cells, {adata.n_vars} genes ({n_hvg_found} HVGs)")
    return adata

# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="Preprocess and convert timepoints in one pass.")
    parser.add_argument("input_h5ad", help="Input .h5ad file")
    parser.add_argument("output_h5ad", help="Output .h5ad file")
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--max_genes", type=int, default=60000)
    parser.add_argument("--min_cells", type=int, default=3)
    parser.add_argument("--n_hvg", type=int, default=3000)

    args = parser.parse_args()

    # Load once
    print("Loading AnnData...")
    adata = sc.read_h5ad(args.input_h5ad)

    # Step 1: Preprocess (QC, HVG, scale)
    adata = preprocess(adata, args)

    adata.write(args.output_h5ad)
    print(f"\nSaved to {args.output_h5ad}")

if __name__ == "__main__":
    main()
