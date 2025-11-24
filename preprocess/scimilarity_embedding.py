#!/usr/bin/env python
import scanpy as sc
import numpy as np
import pandas as pd
from scimilarity.cell_embedding import CellEmbedding
from scimilarity.utils import align_dataset  # 移除lognorm_counts
import sys

def main(input_file, output_file):
    model_path = "/gpfs/flash/home/yr/data/public/SCimilarity/model/model_v1.1/"
    
    print(f"加载模型: {model_path}")
    ce = CellEmbedding(model_path=model_path, use_gpu=False)
    
    print(f"读取数据: {input_file}")
    adata = sc.read(input_file)
    
    print(f"原始数据: {adata.shape}")
    print(f"模型要求基因数: {len(ce.gene_order)}")
    
    # 检查基因重叠
    overlap = len(set(adata.var.index) & set(ce.gene_order))
    print(f"重叠基因数: {overlap}")
    
    # 补全缺失基因
    if overlap < 5000:
        print(f"\n⚠️ 警告: 重叠基因不足({overlap} < 5000)")
        print("将自动补全缺失基因（填充为0）...")
        
        # 转换稀疏矩阵为dense（如果需要）
        if hasattr(adata.X, 'todense'):
            X_dense = adata.X.todense()
        elif hasattr(adata.X, 'toarray'):
            X_dense = adata.X.toarray()
        else:
            X_dense = adata.X
        
        n_cells = adata.n_obs
        n_genes = len(ce.gene_order)
        
        # 创建完整矩阵，缺失基因用0填充（在log1p空间中，0 = log1p(0+1)）
        X_new = np.zeros((n_cells, n_genes), dtype=np.float32)
        
        adata_new = sc.AnnData(
            X=X_new,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=ce.gene_order)
        )
        
        # 复制重叠基因的数据
        gene_map = {gene: i for i, gene in enumerate(ce.gene_order)}
        copied_genes = 0
        
        for gene in adata.var.index:
            if gene in gene_map:
                idx_src = list(adata.var.index).index(gene)
                idx_dst = gene_map[gene]
                X_new[:, idx_dst] = X_dense[:, idx_src]
                copied_genes += 1
        
        print(f"已复制 {copied_genes} 个基因")
        print(f"缺失基因用0填充: {n_genes - copied_genes}")
        adata_aligned = adata_new
        
    else:
        print("基因重叠足够，直接对齐...")
        adata_aligned = align_dataset(adata, ce.gene_order)
    
    # 关键：跳过标准化，因为数据已经是log1p格式
    print(f"\n数据已经是log1p格式，跳过标准化...")
    print(f"数据范围: [{adata_aligned.X.min():.2f}, {adata_aligned.X.max():.2f}]")
    
    # 直接使用aligned数据计算embeddings
    embeddings = ce.get_embeddings(
        adata_aligned.X.astype(np.float32),  # 确保float32
        num_cells=-1,
        buffer_size=10000
    )
    
    # 保存到原始adata
    print(f"保存结果到: {output_file}")
    adata.obsm["X_scimilarity"] = embeddings
    adata.write(output_file)
    
    print(f"✓ 完成！embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python script.py <输入.h5ad> <输出.h5ad>")
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
