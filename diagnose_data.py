#!/usr/bin/env python
"""
数据诊断脚本
快速检查数据质量和预处理状态
"""

import fire
import scanpy as sc
import numpy as np


def diagnose(data_path: str, check_preprocessing: bool = True):
    """
    诊断数据质量
    
    Args:
        data_path: 数据文件路径
        check_preprocessing: 是否测试预处理
    """
    print("=" * 80)
    print("数据诊断报告")
    print("=" * 80)
    print(f"数据路径: {data_path}\n")
    
    # 加载数据
    print("正在加载数据...")
    adata = sc.read_h5ad(data_path)
    print(f"✓ 数据形状: {adata.shape} (细胞数 × 基因数)\n")
    
    # 检查数据类型
    print("数据类型检查:")
    print(f"  X 类型: {type(adata.X)}")
    print(f"  X dtype: {adata.X.dtype if hasattr(adata.X, 'dtype') else 'N/A'}")
    
    # 获取数据用于分析
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
        print(f"  稀疏矩阵: 是 (稀疏率: {1 - adata.X.nnz / (adata.shape[0] * adata.shape[1]):.2%})")
    else:
        X = adata.X
        print(f"  稀疏矩阵: 否")
    print()
    
    # 基本统计
    print("数据统计:")
    print(f"  最小值: {X.min():.6f}")
    print(f"  最大值: {X.max():.6f}")
    print(f"  平均值: {X.mean():.6f}")
    print(f"  标准差: {X.std():.6f}")
    print(f"  中位数: {np.median(X):.6f}")
    print()
    
    # NaN/Inf 检查
    print("数据完整性检查:")
    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()
    has_negative = (X < 0).any()
    
    if has_nan:
        nan_count = np.isnan(X).sum()
        print(f"  ❌ 包含 NaN: {nan_count:,} 个 ({nan_count / X.size:.2%})")
    else:
        print(f"  ✓ 无 NaN")
    
    if has_inf:
        inf_count = np.isinf(X).sum()
        print(f"  ❌ 包含 Inf: {inf_count:,} 个")
    else:
        print(f"  ✓ 无 Inf")
    
    if has_negative:
        neg_count = (X < 0).sum()
        print(f"  ⚠️  包含负值: {neg_count:,} 个 ({neg_count / X.size:.2%})")
    else:
        print(f"  ✓ 无负值")
    print()
    
    # 预处理状态判断
    print("预处理状态判断:")
    max_val = X.max()
    min_val = X.min()
    
    if min_val < 0:
        print(f"  ⚠️  检测到负值（min={min_val:.4f}）！")
        print(f"  判断: 数据已经过标准化/归一化（z-score等）")
        print(f"  ❌ 不能进行 log1p 变换（会产生 NaN）")
        print(f"  建议: 训练时设置 preprocess=False，直接使用当前数据")
    elif max_val < 15 and min_val >= 0:
        print(f"  判断: 数据可能已经 log1p 变换（max={max_val:.2f} < 15, min >= 0)")
        print(f"  建议: 训练时设置 preprocess=False")
    elif max_val > 100:
        print(f"  判断: 数据可能是原始计数（max={max_val:.2f} > 100）")
        print(f"  建议: 训练时设置 preprocess=True（默认）")
    else:
        print(f"  判断: 数据状态不确定（min={min_val:.4f}, max={max_val:.2f}）")
        print(f"  建议: 检查数据或尝试两种设置")
    print()
    
    # 检查 layers
    if hasattr(adata, 'layers') and len(adata.layers) > 0:
        print(f"可用的 layers: {list(adata.layers.keys())}")
        if "counts" in adata.layers:
            print(f"  ✓ 发现 layers['counts']，预处理将使用原始计数")
    else:
        print(f"无 layers（将直接使用 X）")
    print()
    
    # 检查 obs 列
    if hasattr(adata, 'obs'):
        print(f"obs 列: {list(adata.obs.columns)[:10]}")
        if len(adata.obs.columns) > 10:
            print(f"  ... 共 {len(adata.obs.columns)} 列")
    print()
    
    # 测试预处理
    if check_preprocessing:
        print("=" * 80)
        print("测试预处理")
        print("=" * 80)
        
        try:
            from dataset import preprocess_counts
            
            print("正在测试预处理...")
            adata_test = adata.copy()
            adata_processed = preprocess_counts(
                adata_test,
                target_sum=1e4,
                log1p=True,
                verbose=True
            )
            
            if hasattr(adata_processed.X, 'toarray'):
                X_processed = adata_processed.X.toarray()
            else:
                X_processed = adata_processed.X
            
            print(f"\n预处理后统计:")
            print(f"  最小值: {X_processed.min():.6f}")
            print(f"  最大值: {X_processed.max():.6f}")
            print(f"  平均值: {X_processed.mean():.6f}")
            
            if np.isnan(X_processed).any():
                print(f"  ❌ 预处理后包含 NaN！")
            else:
                print(f"  ✓ 预处理成功，无 NaN")
                
        except Exception as e:
            print(f"❌ 预处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(diagnose)

