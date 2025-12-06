# RTF Only 一致性损失 (Consistency Loss)

## 概述

一致性损失是 Rectified Flow 模型的一种正则化技术，通过强制模型在轨迹上保持自洽性来提高采样质量和效率。

---

## 核心思想

在 Rectified Flow 中，我们学习从起点 $x_1$ 到终点 $x_2$ 的速度场 $v(x_t, t)$。理想情况下，从轨迹上任意一点出发，沿着学到的速度场积分，都应该到达正确的位置。

**一致性约束**: 如果模型是完美的，那么：
- 从 $x_{t_1}$ 积分到 $t_2$ 得到的 $\hat{x}_{t_2}$
- 应该等于直接插值得到的 $x_{t_2}$

---

## 数学原理

### 1. 线性插值轨迹

在 Rectified Flow 中，训练轨迹是线性的：

$$x_t = (1-t) \cdot x_1 + t \cdot x_2, \quad t \in [0, 1]$$

目标速度场是常数：

$$v^* = x_2 - x_1$$

### 2. 一致性损失定义

采样两个有序时间点 $t_1 < t_2$：

**Step 1**: 计算真实的中间点
$$x_{t_1} = (1-t_1) \cdot x_1 + t_1 \cdot x_2$$
$$x_{t_2}^{true} = (1-t_2) \cdot x_1 + t_2 \cdot x_2$$

**Step 2**: 用模型从 $x_{t_1}$ 积分到 $t_2$
$$x_{t_2}^{pred} = x_{t_1} + \int_{t_1}^{t_2} v_\theta(x_\tau, \tau) d\tau$$

使用欧拉法近似（n_steps=2）：
$$x_{t_2}^{pred} \approx x_{t_1} + \sum_{k=0}^{n-1} v_\theta(x_k, t_k) \cdot \Delta t$$

**Step 3**: 计算一致性损失
$$\mathcal{L}_{consistency} = \| x_{t_2}^{pred} - x_{t_2}^{true} \|^2$$

### 3. 总损失

$$\mathcal{L}_{total} = \mathcal{L}_{flow} + \lambda \cdot \mathcal{L}_{consistency}$$

其中：
- $\mathcal{L}_{flow}$: 标准的 flow matching 损失
- $\lambda$: 一致性损失权重（默认 0.1）

---

## 代码实现

### RFDirectOnly 类中的实现

```python
def _compute_consistency_loss(
    self,
    x1: torch.Tensor,
    x2: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    n_steps: int = 2,
) -> torch.Tensor:
    """
    计算一致性损失：轨迹上不同时间点预测的终点应该一致
    """
    batch_size = x1.shape[0]
    device = x1.device

    # 采样两个有序时间点 t1 < t2
    t1 = torch.rand(batch_size, device=device) * 0.5  # t1 ∈ [0, 0.5]
    t2 = t1 + torch.rand(batch_size, device=device) * (1.0 - t1)  # t2 ∈ [t1, 1]

    t1_exp = t1.view(batch_size, *([1] * (x1.ndim - 1)))
    t2_exp = t2.view(batch_size, *([1] * (x1.ndim - 1)))

    # 直接插值得到 x_t1 和 x_t2
    x_t1 = (1 - t1_exp) * x1 + t1_exp * x2
    x_t2_true = (1 - t2_exp) * x1 + t2_exp * x2

    # 用模型从 x_t1 积分到 t2
    dt = (t2 - t1) / n_steps
    x_current = x_t1.clone()

    for step in range(n_steps):
        t_current = t1 + step * dt
        v = self.backbone(x_current, t_current, cond)
        x_current = x_current + v * dt.view(-1, *([1] * (x1.ndim - 1)))

    x_t2_pred = x_current

    # 一致性损失
    consistency_loss = F.mse_loss(x_t2_pred, x_t2_true)

    return consistency_loss
```

### 时间采样策略

```
时间轴:  0 -------- t1 -------- t2 -------- 1
              [0, 0.5]    [t1, 1.0]
                 ↓            ↓
              起点区域     终点区域
```

- $t_1 \sim U(0, 0.5)$: 在前半段采样起点
- $t_2 \sim U(t_1, 1.0)$: 在 $t_1$ 之后采样终点
- 确保 $t_1 < t_2$，且时间间隔有一定长度

---

## 可视化理解

```
                    真实轨迹 (线性插值)
    x1 ─────────────●─────────────────●─────────────── x2
                   x_t1              x_t2_true
                    │
                    │  模型积分 (n_steps=2)
                    │     ↓
                    ●────►●────►●
                   x_t1        x_t2_pred

    一致性损失 = ||x_t2_pred - x_t2_true||²
```

如果模型完美学习了速度场，则：
- $x_{t_2}^{pred} = x_{t_2}^{true}$
- 一致性损失 = 0

---

## 配置使用

### YAML 配置

```yaml
# config/rtf_only.yaml
training:
  use_consistency: true       # 启用一致性损失
  consistency_weight: 0.1     # 一致性损失权重
```

### 命令行覆盖

```bash
python train_rtf_only.py \
    --training__use_consistency=true \
    --training__consistency_weight=0.1
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_consistency` | bool | false | 是否启用一致性损失 |
| `consistency_weight` | float | 0.1 | 一致性损失权重 $\lambda$ |

---

## 训练日志

启用一致性损失后，训练日志会显示：

```
train_loss: 0.0850    # 总损失
flow_loss: 0.0800     # flow matching 损失
consistency_loss: 0.0500  # 一致性损失 (乘以权重前)
```

实际总损失计算：
```
train_loss = flow_loss + consistency_weight × consistency_loss
           = 0.0800 + 0.1 × 0.0500
           = 0.0850
```

---

## 理论分析

### 为什么一致性损失有效？

1. **减少累积误差**: 单步预测误差会在多步采样中累积，一致性损失直接惩罚多步积分误差

2. **平滑速度场**: 强制轨迹上各点的预测一致，使速度场更加平滑

3. **提高采样效率**: 减少采样步数时仍能保持质量

### 与其他方法的对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **Flow Matching** | 直接拟合速度场 | 简单高效 | 多步误差累积 |
| **Consistency Loss** | 轨迹自洽约束 | 减少累积误差 | 额外计算开销 |
| **Distillation** | 教师-学生蒸馏 | 可大幅减少步数 | 需要预训练教师 |

---

## Inversion 模式的一致性损失

对于 Inversion 模式（x1 → noise → x2），一致性损失分别应用于两个方向：

```python
# 方向 1: x1 → noise
consistency_loss1 = self._compute_consistency_loss(x1, noise1, cond1)

# 方向 2: x2 → noise
consistency_loss2 = self._compute_consistency_loss(x2, noise2, cond2)

# 平均
consistency_loss = (consistency_loss1 + consistency_loss2) / 2
```

---

## 超参数调优建议

### consistency_weight 选择

| 权重值 | 效果 |
|--------|------|
| 0.01 | 轻微正则化，几乎不影响主损失 |
| **0.1** | 推荐值，平衡主损失和一致性 |
| 0.5 | 强正则化，可能影响收敛速度 |
| 1.0 | 与主损失同等权重，需谨慎使用 |

### 训练策略

1. **渐进式启用**: 先不用一致性损失训练几个 epoch，再启用
2. **权重衰减**: 随训练进行逐渐降低一致性权重
3. **监控平衡**: 确保 flow_loss 和 consistency_loss 在同一数量级

---

## 实验效果

### 预期改进

- 减少采样步数时的质量下降
- 轨迹更加平滑
- 远距离预测更准确

### 何时使用

✅ 推荐使用：
- 需要快速采样（少步数）
- 长时间跨度预测
- 模型已基本收敛，需要精调

❌ 不推荐使用：
- 训练初期（可能干扰收敛）
- 计算资源有限（增加约 50% 计算量）
- 短时间跨度预测（收益不明显）

---

## 参考文献

1. Liu et al. "Flow Matching for Generative Modeling" (2022)
2. Song et al. "Consistency Models" (2023)
3. Lipman et al. "Flow Matching for Generative Modeling" (2023)
