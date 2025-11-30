import scanpy as sc
import pandas as pd
import sys

h5ad_path = "/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/SRX21870170.h5ad"
vocab_path = "data_info/gene_order.tsv"

print(f"Reading h5ad: {h5ad_path}")
try:
    adata = sc.read_h5ad(h5ad_path)
    print(f"Adata shape: {adata.shape}")
    print(f"Adata var_names[:5]: {adata.var_names[:5].tolist()}")
except Exception as e:
    print(f"Error reading h5ad: {e}")
    sys.exit(1)

print(f"Reading vocab: {vocab_path}")
try:
    with open(vocab_path, 'r') as f:
        vocab = [line.strip() for line in f if line.strip()]
    print(f"Vocab size: {len(vocab)}")
    print(f"Vocab[:5]: {vocab[:5]}")
except Exception as e:
    print(f"Error reading vocab: {e}")
    sys.exit(1)

overlap = set(adata.var_names) & set(vocab)
print(f"Overlap size: {len(overlap)}")

if len(overlap) == 0:
    print("WARNING: NO OVERLAP! This explains the zero data.")
    # Check if maybe var has a column with symbols?
    print("Checking adata.var columns:")
    print(adata.var.head())
else:
    print(f"Overlap ratio: {len(overlap)/len(vocab):.2%}")

