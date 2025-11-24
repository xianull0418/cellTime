import scanpy as sc
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert TEDD counts + meta to .h5ad with dataset-level metadata.")
    parser.add_argument("counts_file", help="Path to *.counts.csv.gz")
    parser.add_argument("meta_file", help="Path to cell-level *.meta.csv")
    parser.add_argument("output_file", help="Output .h5ad file")
    parser.add_argument("--table", default="tedd_datasets_table.csv", help="Path to tedd_datasets_table.csv")
    args = parser.parse_args()

    # 提取ID
    tedd_id = os.path.basename(args.counts_file).replace(".counts.csv.gz", "")
    
    # 读取dataset元数据
    table = pd.read_csv(args.table, index_col="ID", dtype=str, keep_default_na=False)
    dataset_meta = table.loc[tedd_id].drop("Download").to_dict()

    # 读取并处理数据
    adata = sc.read_csv(args.counts_file, first_column_names=True).T
    adata.obs = pd.read_csv(args.meta_file, index_col=0)
    adata.uns.update(dataset_meta)
    
    # 保存
    sc.write(args.output_file, adata)
    print(f"Saved: {args.output_file}")

if __name__ == "__main__":
    main()
