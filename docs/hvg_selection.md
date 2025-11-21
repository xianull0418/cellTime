# 高变基因自动选择功能

## 概述

当数据集的基因数超过模型配置的 `n_genes` 时，系统会自动选择高变基因（Highly Variable Genes, HVGs）以匹配模型的输入维度。

## 更新内容

### 1. 数据集层面 (`dataset/cell_dataset.py`)

#### 新增函数
- `_select_highly_variable_genes()`: 使用 Scanpy 的 Seurat v3 方法选择高变基因

#### 更新的类
所有数据集类都新增了 `max_genes` 参数：

- `StaticCellDataset` (用于 AE 训练)
- `TemporalCellDataset` (用于 RTF 训练)
- `MultiCellDataset` (用于集合编码器)

**新增参数:**
```python
max_genes: Optional[int] = None
```

**功能:**
- 如果数据集基因数 > `max_genes`，自动选择前 `max_genes` 个高变基因
- 使用 Scanpy 的 `highly_variable_genes` 方法 (flavor='seurat_v3')
- 如果 seurat_v3 失败，回退到默认的 seurat 方法

### 2. AE 训练 (`models/ae.py`)

**更新的方法:**
- `AESystem.setup()`: 在创建数据集时传递 `max_genes` 参数

**行为:**
```python
# 如果配置文件中 n_genes = 3000，但数据集有 14270 个基因
# 系统会自动选择 3000 个高变基因
max_genes = self.cfg.model.n_genes if self.cfg.model.n_genes > 0 else None

full_dataset = StaticCellDataset(
    self.cfg.data.data_path,
    max_genes=max_genes,  # 传递 max_genes
    verbose=True,
    seed=42,
)
```

### 3. RTF 训练 (`train_rtf.py`, `models/rtf.py`)

**train_rtf.py 更新:**
- 从 AE checkpoint 中读取 `n_genes`
- 自动同步到 RTF 配置中

```python
ae_n_genes = ae_system.cfg.model.n_genes
cfg.model.n_genes = ae_n_genes
print(f"✓ RTF 数据集将使用 AE 的基因数: {ae_n_genes}")
```

**RTFSystem.setup() 更新:**
- 在创建 `TemporalCellDataset` 时传递 `max_genes`
- 确保 RTF 训练使用与 AE 相同的基因集

### 4. 配置文件更新 (`config/rtf.yaml`)

新增字段:
```yaml
model:
  n_genes: null  # 基因数（将从 AE 自动同步）
```

## 使用示例

### 场景 1: AE 训练时自动选择高变基因

```bash
# 配置文件中设置 n_genes: 3000
# 数据集有 14270 个基因
python train_ae.py --config_path=config/ae_scimilarity.yaml --data_path=data.h5ad

# 输出:
# 数据加载完成，维度: (2989, 14270)
# 基因数 (14270) 超过上限 (3000)，自动选择 3000 个高变基因...
# 已选择 3000 个高变基因
# 高变基因选择完成，新维度: (2989, 3000)
# 从 layer 'log1p' 读取表达数据
# StaticCellDataset 初始化完成:
#   - 细胞数: 2989
#   - 基因数: 3000
```

### 场景 2: RTF 训练时自动匹配 AE 的基因集

```bash
# AE 使用了 3000 个高变基因
python train_rtf.py \
  --ae_checkpoint=output/ae/checkpoints/last.ckpt \
  --data_path=temporal_data.h5ad

# 输出:
# AE 潜空间维度: 128
# AE 基因数: 3000
# ✓ RTF 数据集将使用 AE 的基因数: 3000
# 基因数 (14270) 超过上限 (3000)，自动选择 3000 个高变基因...
# 训练数据集大小: 2690
# 基因数量: 3000
```

## 技术细节

### 高变基因选择方法

使用 Scanpy 的 `highly_variable_genes` 函数：

1. **优先使用 Seurat v3 方法:**
   ```python
   sc.pp.highly_variable_genes(
       adata,
       n_top_genes=n_top_genes,
       flavor='seurat_v3',
       layer='counts',
       subset=False,
   )
   ```

2. **回退到默认 Seurat 方法:**
   如果 Seurat v3 失败（例如，没有 counts layer），使用：
   ```python
   sc.pp.highly_variable_genes(
       adata,
       n_top_genes=n_top_genes,
       flavor='seurat',
       subset=False,
   )
   ```

3. **筛选:**
   ```python
   adata = adata[:, adata.var['highly_variable']].copy()
   ```

### Layer 选择

数据集默认从 `adata.layers['log1p']` 读取数据：
- 如果 layer 存在，使用该 layer
- 如果 layer 不存在，回退到 `adata.X`
- 高变基因选择在 layer 选择之前进行

## 注意事项

1. **基因选择一致性:** RTF 训练必须使用与 AE 训练相同的基因集，系统会自动确保这一点

2. **数据预处理:** 高变基因选择最好在 log1p 转换前的 counts 数据上进行

3. **可重复性:** 高变基因选择是确定性的（基于基因的变异性排序）

4. **性能:** 对于大型数据集（>10万细胞），高变基因计算可能需要几分钟

## 禁用高变基因选择

如果需要使用所有基因，可以：

1. **配置文件中设置:**
   ```yaml
   model:
     n_genes: 0  # 或者设置为 null
   ```

2. **手动指定:**
   ```python
   dataset = StaticCellDataset(
       data_path,
       max_genes=None,  # 不限制基因数
       verbose=True,
   )
   ```

## 错误处理

如果基因数不匹配但没有自动选择，会出现以下错误：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x14270 and 3000x1024)
```

解决方法：确保配置文件中正确设置了 `n_genes`。

