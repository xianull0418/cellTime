#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import re
import sys
from scipy.stats import median_abs_deviation

# ============ 时间解析部分 ============
from format_time_h5ad import process_anndata_timepoints

# ============ QC 预处理部分 ============
from preprocess_h5ad_v2 import preprocess
#from tmp import preprocess

# ============ 主函数 ============

def main():
    parser = argparse.ArgumentParser(description="Preprocess and convert timepoints in one pass.")
    parser.add_argument("input_h5ad", help="Input .h5ad file")
    parser.add_argument("output_h5ad", help="Output .h5ad file")
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--max_genes", type=int, default=60000)
    parser.add_argument("--min_cells", type=int, default=3)
    parser.add_argument("--n_hvg", type=int, default=3000)
    parser.add_argument("--time_col", type=str, default="Timepoint", help="Column name for timepoint in adata.obs")

    args = parser.parse_args()

    # Load once
    print("Loading AnnData...")
    adata = sc.read_h5ad(args.input_h5ad)

    # Step 1: Preprocess (QC, HVG, scale)
    adata = preprocess(adata, args)

    # Step 2: Time conversion (requires Species in uns)
    print("Converting timepoints to days post-fertilization...")
    adata = process_anndata_timepoints(adata, time_col=args.time_col, new_col='time')

    adata.write(args.output_h5ad)
    print(f"\nSaved to {args.output_h5ad}")

if __name__ == "__main__":
    main()
