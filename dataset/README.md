# 数据预处理说明

## 概述

本项目的数据预处理流程参考 **scimilarity** 的标准流程，确保与预训练模型的数据格式一致。

## 预处理流程

### 1. 归一化（Normalize Total）
- **目标**: 每个细胞的总计数归一化到固定值
- **默认值**: `target_sum = 10,000` (1e4)
- **作用**: 消除测序深度差异，使不同细胞可比

### 2. Log1p 变换
- **公式**: `log(1 + x)`
- **作用**: 压缩数值范围，减少异常值影响
- **结果范围**: 通常在 [0, 10] 左右

## 使用方法

### 自动预处理（推荐）

所有Dataset类默认会自动进行预处理：

```python
from dataset import StaticCellDataset, TemporalCellDataset

# AE训练数据集（自动预处理）
train_dataset = StaticCellDataset(
    data="path/to/data.h5ad",
    preprocess=True,  # 默认True
    target_sum=1e4,   # 默认1e4
    verbose=True
)

# RTF训练数据集（自动预处理）
temporal_dataset = TemporalCellDataset(
    data="path/to/temporal_data.h5ad",
    preprocess=True,  # 默认True
    target_sum=1e4,   # 默认1e4
    verbose=True
)
```

### 手动预处理

如果需要单独预处理数据：

```python
from dataset.cell_dataset import preprocess_counts
import scanpy as sc

# 加载数据
adata = sc.read_h5ad("path/to/data.h5ad")

# 预处理
adata = preprocess_counts(
    adata,
    target_sum=1e4,
    log1p=True,
    verbose=True
)
```

### 跳过预处理

如果数据已经预处理过，可以跳过：

```python
dataset = StaticCellDataset(
    data="preprocessed_data.h5ad",
    preprocess=False,  # 跳过预处理
    verbose=True
)
```

## 数据格式要求

### 原始计数数据

如果你的数据是原始计数（raw counts）：
- 存储在 `adata.X` 或 `adata.layers["counts"]`
- 数值范围通常在 [0, 10000+]
- **必须**进行预处理：`preprocess=True`

### 已归一化数据

如果数据已经log1p变换过：
- 数值范围通常在 [0, 10] 左右
- **跳过**预处理：`preprocess=False`
- 或者让自动检测判断（会打印警告）

## 验证预处理

训练时会自动打印诊断信息：

```
[诊断 Batch 0]
  原始空间 x_cur: min=0.0000, max=9.2145, mean=1.4532, std=1.2345
  潜空间 z_cur: min=-2.1234, max=3.4567, mean=0.0123, std=0.9876
```

**正常范围**：
- 原始空间（log1p后）: [0, 10]
- 潜空间（AE编码后）: [-5, 5] 左右

## 与 scimilarity 的兼容性

本项目的预处理流程与 scimilarity 完全一致：

```python
# scimilarity 流程
from scimilarity.utils import lognorm_counts

adata = lognorm_counts(adata)
# -> normalize_total(target_sum=1e4) + log1p()

# 我们的流程
from dataset.cell_dataset import preprocess_counts

adata = preprocess_counts(adata, target_sum=1e4, log1p=True)
# -> 完全相同！
```

## 常见问题

### Q: 我的数据没有 `layers["counts"]` 怎么办？
A: 预处理会自动使用 `adata.X`，不影响使用。

### Q: 如何判断数据是否已经预处理？
A: 查看数据的最大值：
- 如果 `max > 100`：可能是原始计数，需要预处理
- 如果 `max < 20`：可能已经log变换，可跳过预处理

### Q: 可以使用其他 target_sum 吗？
A: 可以，但推荐使用 1e4 以保持与 scimilarity 预训练模型的一致性。

### Q: 预处理对性能有什么影响？
A: 正确的预处理可以：
- 提高训练稳定性
- 加快收敛速度
- 提升模型性能
- 与预训练模型兼容

## 参考

- scimilarity: https://github.com/Genentech/scimilarity
- scanpy preprocessing: https://scanpy.readthedocs.io/en/stable/api/preprocessing.html

