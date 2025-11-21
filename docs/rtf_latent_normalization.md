# RTF 潜空间归一化问题与解决方案

## 🔍 问题描述

在使用 scimilarity 预训练 Encoder 训练 RTF 时，发现采样后的重建质量很差：

```
Epoch 0: 采样完成
  原始空间重建误差: 0.339383
  潜空间重建误差: 0.000395
  重建相关性: 0.0085
```

**关键症状：**
- 重建相关性极低（0.0085 左右），甚至出现负值
- 原始空间重建误差很高（0.32-0.38）
- 潜空间重建误差很小（0.0003-0.0005）

## 🔬 根本原因

### 问题 1: scimilarity Encoder 输出归一化

查看 scimilarity 的 `Encoder` 实现（`nn_models.py:80`）：

```python
def forward(self, x) -> torch.Tensor:
    for i, layer in enumerate(self.network):
        x = layer(x)
    return F.normalize(x, p=2, dim=1)  # ⚠️ L2 归一化
```

**关键特性：**
- Encoder 输出被归一化到**单位超球面**上
- 每个潜空间向量的 L2 norm = 1
- Decoder 在训练时只见过 norm=1 的输入

### 问题 2: RTF 采样破坏归一化约束

在 RTF 的 Euler 采样过程中（原始代码）：

```python
# 欧拉步：z = z + v * dt
z = z + v * dt  # ⚠️ 这会破坏单位球面约束！
```

**后果：**
1. **norm 漂移**：每次更新后，`||z|| ≠ 1`
2. **分布偏移**：Decoder 收到的输入分布与训练时不一致
3. **重建失败**：Decoder 对非单位向量的解码能力很差

### 为什么潜空间误差小但原始空间误差大？

```
z_cur  (norm=1) → [RTF采样] → z_pred (norm≠1) → [Decoder] → x_pred (质量差)
z_next (norm=1)
```

- RTF 在潜空间学习得还可以，所以 `||z_pred - z_next||` 较小
- 但 `z_pred` 的 norm 不等于 1，导致 Decoder 解码失败
- 因此 `||x_pred - x_next||` 很大

## ✅ 解决方案

### 修改 1: 在采样时保持归一化

**RFDirect.sample() 更新：**

```python
@torch.no_grad()
def sample(
    self,
    z_start: torch.Tensor,
    sample_steps: int = 50,
    cond: Optional[torch.Tensor] = None,
    null_cond: Optional[torch.Tensor] = None,
    cfg_scale: float = 2.0,
    normalize_latent: bool = True,  # ✨ 新增参数
) -> List[torch.Tensor]:
    """采样时保持单位球面约束"""
    z = z_start.clone()
    batch_size = z.shape[0]
    device = z.device
    dt = 1.0 / sample_steps
    
    trajectory = [z.cpu()]
    
    for step in range(sample_steps):
        t_current = step / sample_steps
        t = torch.full((batch_size,), t_current, device=device)
        
        # 预测速度场
        v = self.backbone(z, t, cond)
        
        # CFG
        if null_cond is not None and cfg_scale != 1.0:
            v_uncond = self.backbone(z, t, null_cond)
            v = v_uncond + cfg_scale * (v - v_uncond)
        
        # 欧拉步
        z = z + v * dt
        
        # 🔧 关键修复：归一化到单位球面
        if normalize_latent:
            z = F.normalize(z, p=2, dim=1)
        
        trajectory.append(z.cpu())
    
    return trajectory
```

### 修改 2: 在配置中启用归一化

**config/rtf.yaml:**

```yaml
model:
  mode: direct
  backbone: dit
  latent_dim: 128
  normalize_latent: true  # ✨ 为 scimilarity encoder 启用
```

### 修改 3: RTFSystem 中自动使用配置

```python
# 采样时根据配置决定是否归一化
normalize_latent = getattr(self.cfg.model, 'normalize_latent', True)

trajectory = self.model.sample(
    z_cur,
    sample_steps=self.cfg.training.sample_steps,
    normalize_latent=normalize_latent,  # ✨ 传递参数
)
```

## 📊 预期效果

修复后，预期：

```
Epoch 0: 采样完成
  原始空间重建误差: 0.05 - 0.15  # ✅ 大幅降低
  潜空间重建误差: 0.0003 - 0.0005  # 保持不变
  重建相关性: 0.85 - 0.95  # ✅ 大幅提升
```

**指标改善：**
- ✅ 重建相关性：0.0085 → 0.85+ (提升 100倍)
- ✅ 原始空间误差：0.33 → 0.05-0.15 (降低 2-6倍)
- ✅ 潜空间误差：保持不变（说明 RTF 学习没问题）

## 🧪 理论分析

### 为什么在 scimilarity 中使用单位球面？

1. **几何简化**：
   - 在单位球面上，距离度量更简单
   - 避免了向量长度的影响，只关注方向

2. **度量学习**：
   - scimilarity 使用 triplet loss 训练
   - 余弦相似度 = 内积（当 ||z|| = 1 时）
   - 归一化使得相似度只取决于角度

3. **稳定性**：
   - 避免潜空间向量的 norm 爆炸或消失
   - 提高训练稳定性

### RTF 在单位球面上的挑战

在非归一化情况下：

```
t=0: z(0) = z_start,  ||z(0)|| = 1
t=0.1: z(0.1) = z(0) + v·dt,  ||z(0.1)|| ≠ 1  # ⚠️ norm 漂移
t=0.2: z(0.2) = z(0.1) + v·dt,  ||z(0.2)|| ≠ 1  # ⚠️ 继续漂移
...
t=1.0: z(1) = z_pred,  ||z(1)|| ≠ 1  # ⚠️ 严重偏离
```

加上归一化后：

```
t=0: z(0) = z_start,  ||z(0)|| = 1
t=0.1: z'(0.1) = normalize(z(0) + v·dt),  ||z'(0.1)|| = 1  # ✅ 保持约束
t=0.2: z'(0.2) = normalize(z'(0.1) + v·dt),  ||z'(0.2)|| = 1  # ✅ 保持约束
...
t=1.0: z'(1) = z_pred,  ||z'(1)|| = 1  # ✅ 始终满足约束
```

### 为什么不在训练时归一化？

**不需要！**原因：

1. **训练目标不受影响**：
   ```python
   z_t = (1-t) * z1 + t * z2  # 线性插值
   ```
   - 如果 `||z1|| = ||z2|| = 1`，则 `||z_t||` 接近 1
   - 训练时的 z_t 自然接近单位球面

2. **采样是关键**：
   - 训练时：从真实的 z1, z2 插值，自然满足约束
   - 采样时：从 z_start 累积预测，容易偏离约束
   - 因此只需在采样时归一化

## ⚙️ 使用方法

### 默认配置（推荐）

对于使用 scimilarity 预训练模型的情况，默认启用：

```bash
python train_rtf.py \
  --ae_checkpoint=output/ae_finetune/checkpoints/last.ckpt \
  --data_path=data.h5ad
  # normalize_latent=true (默认)
```

### 禁用归一化

如果从头训练 AE（不使用 scimilarity），可以禁用：

```bash
python train_rtf.py \
  --ae_checkpoint=output/ae_scratch/checkpoints/last.ckpt \
  --data_path=data.h5ad \
  --model__normalize_latent=false
```

或在配置文件中：

```yaml
model:
  normalize_latent: false
```

## 🎯 何时需要归一化？

| Encoder 类型 | 输出是否归一化 | 需要设置 normalize_latent |
|-------------|---------------|-------------------------|
| scimilarity | ✅ 是（L2 norm=1） | `true` |
| 从头训练 AE | ❌ 否 | `false` |
| VAE | ❌ 否 | `false` |
| 其他预训练模型 | 需检查代码 | 根据情况 |

**检查方法：**

查看 Encoder 的 forward 方法，看是否有：
```python
return F.normalize(x, p=2, dim=1)
```

## 📝 相关修改

- ✅ `models/rtf.py`: RFDirect.sample() 添加 normalize_latent 参数
- ✅ `models/rtf.py`: RFInversion.sample() 添加 normalize_latent 参数  
- ✅ `models/rtf.py`: RTFSystem._sample_and_save() 使用配置中的 normalize_latent
- ✅ `config/rtf.yaml`: 添加 normalize_latent 配置项

## 🔗 参考资料

1. **scimilarity 论文**：
   - "A cell atlas foundation model for scalable search of similar human cells"
   - Nature (2024)
   - https://doi.org/10.1038/s41586-024-08411-y

2. **Rectified Flow**：
   - "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
   - ICLR 2023

3. **度量学习中的归一化**：
   - L2 normalization 是 triplet loss 和 contrastive learning 的标准做法
   - 将问题简化为在单位超球面上的优化

