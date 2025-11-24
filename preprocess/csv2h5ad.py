import scanpy as sc
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert counts + meta to .h5ad with metadata.")
    parser.add_argument("counts_file", help="Path to *.counts.csv.gz")
    parser.add_argument("cell_meta", help="Path to cell-level *.meta.csv")
    parser.add_argument("gene_meta", help="Path to gene-level *.meta.csv")
    parser.add_argument("output_file", help="Output .h5ad file")
    args = parser.parse_args()

    # 读取dataset元数据


    # 读取并处理数据
    adata = sc.read_csv(args.counts_file, first_column_names=True).T
    adata.obs = pd.read_csv(args.cell_meta, index_col=0)
    adata.vars = pd.read_csv(args.gene_meta, index_col=0)
    
    # 保存
    sc.write(args.output_file, adata)
    print(f"Saved: {args.output_file}")

if __name__ == "__main__":
    main()
