# RTF Only 模型 Benchmark 结果

## 概述

本文档描述 RTF Only 模型的评估指标和可视化结果。

- **模型**: RTF Only (Rectified Flow，无 AutoEncoder)
- **数据集**: CellRank 时序单细胞数据
- **评估日期**: 2024-12-05
- **结果目录**: `benchmarks/results/rtf_only_cellrank/`

---

## 评估指标

### 核心指标

| 指标 | 值 |
|------|-----|
| 评估细胞数 | 2,989 |
| **平均相关性 (Correlation)** | **0.8416 ± 0.0663** |
| 相关性中位数 | 0.8376 |
| 平均 MSE | 0.0894 ± 0.0364 |
| 相关性 > 0.8 | 2,519 (84.3%) |
| 相关性 > 0.9 | 323 (10.8%) |

### 分时间点指标

| 时间点 | 样本数 | 平均相关性 |
|--------|--------|------------|
| t=0.0 | 50 | 0.835 |
| t=1.0 | 50 | 0.846 |
| t=2.0 | 50 | 0.833 |
| t=4.0 | 50 | 0.800 |
| t=9.0 | 50 | 0.781 |

**观察**: 后期时间点预测难度略大，相关性有下降趋势。

---

## 输出文件说明

### 1. `metrics.csv` - 原始指标数据

每个细胞的预测评估结果：

```csv
Correlation,MSE
0.7834745,0.10686726
0.85775876,0.07978967
...
```

- **Correlation**: 预测表达向量与真实表达向量的 Pearson 相关系数
- **MSE**: 均方误差 (Mean Squared Error)

---

### 2. `correlation_dist.png` - 相关性分布图

![correlation_dist](../benchmarks/results/rtf_only_cellrank/correlation_dist.png)

**描述**:
- 展示所有评估细胞的预测相关性分布
- X 轴: Pearson 相关系数
- Y 轴: 细胞数量

**解读**:
- 主峰位于 0.80-0.85，说明大部分预测质量良好
- 右侧尖峰在 ~1.0 附近，表示部分细胞预测几乎完美
- 绝大多数细胞相关性 > 0.7

---

### 3. `trajectory_correlation_summary.png` - 相关性统计汇总

![trajectory_correlation_summary](../benchmarks/results/rtf_only_cellrank/trajectory_correlation_summary.png)

包含三个子图：

| 子图 | 描述 |
|------|------|
| **左: 箱线图** | 各时间点的相关性分布，展示中位数、四分位数和异常值 |
| **中: 直方图** | 可视化采样细胞的相关性分布，红色虚线标注均值 (0.819) |
| **右: 趋势图** | 平均相关性随时间的变化趋势，阴影区域表示标准差 |

**解读**:
- 各时间点中位数均在 0.8 以上
- 随时间推移，预测精度略有下降
- 部分异常值（离群点）相关性较低

---

### 4. `trajectory_pca_2d.png` - 2D PCA 空间轨迹图

![trajectory_pca_2d](../benchmarks/results/rtf_only_cellrank/trajectory_pca_2d.png)

**描述**: 在 PCA 降维后的 2D 空间展示细胞轨迹预测

**图例**:
| 符号 | 含义 |
|------|------|
| ○ (圆点) | 起始细胞 (时刻 t) |
| □ (方块) | 真实的下一时刻细胞 (t+1) |
| △ (三角) | 模型预测的下一时刻细胞 |
| → (红色箭头) | 预测方向 (起始 → 预测) |
| -- (虚线) | 真实方向 (起始 → 真实) |
| 颜色条 | 时间点 (紫色=0 → 黄色=9) |

**解读**:
- 预测三角形与真实方块越接近，预测越准确
- 红色箭头与虚线方向越一致，预测方向越正确
- 不同颜色代表不同发育时间阶段

---

### 5. `trajectory_per_timepoint.png` - 分时间点轨迹对比

![trajectory_per_timepoint](../benchmarks/results/rtf_only_cellrank/trajectory_per_timepoint.png)

**描述**: 每个时间点单独展示预测效果

**图例**:
| 符号 | 颜色 | 含义 |
|------|------|------|
| ● | 蓝色 | 起始细胞 (Start) |
| ■ | 绿色 | 真实下一时刻 (True Next) |
| ▲ | 红色 | 预测结果 (Predicted) |

**标题格式**: `t=X (n=样本数, r=平均相关性)`

**解读**:
- 每个子图展示特定时间点的 50 个采样细胞
- 红色三角形与绿色方块越接近，预测越准确
- 可以直观比较不同时间点的预测质量

---

### 6. `trajectory_vector_field.png` - 向量场可视化

![trajectory_vector_field](../benchmarks/results/rtf_only_cellrank/trajectory_vector_field.png)

**描述**: 以向量场形式同时展示预测方向和真实方向

**图例**:
| 箭头颜色 | 含义 |
|----------|------|
| 红色 | 模型预测的位移方向 (Predicted Direction) |
| 蓝色 | 真实的位移方向 (True Direction) |

**点颜色**: 表示时间点 (紫色=0 → 黄色=9)

**解读**:
- 红色和蓝色箭头方向越一致，预测越准确
- 可以观察不同区域的预测质量
- 有助于发现模型在特定区域的预测偏差

---

## 运行 Benchmark

```bash
# 使用默认配置
./benchmark_rtf.sh

# 指定检查点
./benchmark_rtf.sh output/rtf_only_experiment/checkpoints/best.ckpt

# 指定所有参数
./benchmark_rtf.sh checkpoint.ckpt data.h5ad output_dir
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CKPT_PATH` | `output/rtf_only_experiment_cellrank/checkpoints/last.ckpt` | 模型检查点 |
| `DATA_PATH` | (使用训练配置) | 数据路径 |
| `OUTPUT_DIR` | `benchmarks/results/rtf_only_cellrank` | 输出目录 |
| `BATCH_SIZE` | 100 | 批次大小 |
| `SAMPLE_STEPS` | 20 | ODE 采样步数 |
| `VIS_CELLS_PER_TIME` | 50 | 每个时间点可视化的细胞数 |

---

## 指标计算方法

### Pearson 相关系数

对于预测向量 $\hat{x}$ 和真实向量 $x$:

$$r = \frac{\sum_{i}(x_i - \bar{x})(\hat{x}_i - \bar{\hat{x}})}{\sqrt{\sum_{i}(x_i - \bar{x})^2} \sqrt{\sum_{i}(\hat{x}_i - \bar{\hat{x}})^2}}$$

- 范围: [-1, 1]
- 1 表示完美正相关
- 0.8+ 通常认为是良好的预测

### MSE (均方误差)

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2$$

- 范围: [0, +∞)
- 越小越好

---

## 结论

当前 RTF Only 模型在 CellRank 数据集上表现良好：

1. **整体相关性高**: 平均 0.84，84.3% 的细胞相关性 > 0.8
2. **时间稳定性**: 各时间点预测质量相近，后期略有下降
3. **方向预测准确**: 向量场可视化显示预测方向与真实方向基本一致

**潜在改进方向**:
- 对后期时间点增加训练权重
- 尝试一致性损失 (`use_consistency: true`)
- 调整采样步数以平衡速度和精度
