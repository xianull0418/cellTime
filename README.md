# scTFM
1. **阶段 1 - Autoencoder (AE)**：将原始单细胞基因表达数据（n_genes 维）编码到低维潜空间（latent_dim 维）
2. **阶段 2 - Rectified Flow (RTF)**：在 AE 潜空间训练流模型，使用 DiT 或 MLP 作为骨干网络预测速度场

```
cellTime/
├── config/                          # 配置文件
│   ├── ae.yaml                      # AE 配置
│   ├── rtf.yaml                     # RTF 配置（支持 direct/inversion 模式）
│   └── backbones/
│       ├── dit.yaml                 # DiT 骨干网络配置
│       ├── mlp.yaml                 # MLP 骨干网络配置
│       └── ……                       # 可以实现更多backbone
├── models/                          # 模型实现
│   ├── __init__.py
│   ├── ae.py                        # Autoencoder（整合 Encoder + Decoder）
│   ├── rtf.py                       # Rectified Flow（RFDirect + RFInversion）
│   ├── utils.py                     # 工具函数（时间编码、相关系数等）
│   └── backbones/
│       ├── __init__.py
│       ├── base.py                  # 骨干网络抽象基类
│       ├── dit.py                   # DiT（适配单细胞潜空间）
│       └── mlp.py                   # MLP 速度场预测器
│
├── dataset/                         # 数据集模块
│   ├── __init__.py
│   └── cell_dataset.py              # 单细胞数据集
│       ├── StaticCellDataset        # 静态数据（AE 训练）
│       ├── TemporalCellDataset      # 时序数据（RTF 训练）
│       └── MultiCellDataset         # 多细胞数据
│
├── train_ae.py                      # AE 训练脚本
├── train_rtf.py                     # RTF 训练脚本
└── examples/
    └── quick_start.sh               # ✅ 快速开始脚本
```

```
原始数据 (n_genes)
    ↓
[AE Encoder]
    ↓
潜空间 z (latent_dim)  ← 【RTF 在这里训练】
    ↓
[RTF + DiT/MLP] → 预测速度场 v
    ↓
欧拉采样生成 z2
    ↓
[AE Decoder]
    ↓
重建数据 (n_genes)
```
## 模型训练
### 1. 训练 Autoencoder

```bash
# 基础训练
python train_ae.py \
  --data.data_path=data/your_data.h5ad \
  --model.n_genes=3000 \
  --model.latent_dim=256 \
  --training.max_epochs=100

# 自定义配置
python train_ae.py \
  --config_path=config/ae.yaml \
  --data.data_path=data/your_data.h5ad \
  --model.latent_dim=128 \
  --model.hidden_dim='[512,256]' \
  --training.batch_size=512 \
  --training.learning_rate=5e-4 \
  --logging.output_dir=output/ae_custom
```

训练完成后，checkpoint 保存在 `output/ae/checkpoints/`。

### 2. 训练 Rectified Flow

#### 方案 A：Direct 模式 + DiT 骨干网络

```bash
python train_rtf.py \
  --ae_checkpoint=output/ae/checkpoints/last.ckpt \
  --data.data_path=data/temporal_data.h5ad \
  --model.mode=direct \
  --model.backbone=dit \
  --model.latent_dim=256 \
  --training.max_epochs=100
```

#### 方案 B：Inversion 模式 + MLP 骨干网络

```bash
python train_rtf.py \
  --ae_checkpoint=output/ae/checkpoints/last.ckpt \
  --data.data_path=data/temporal_data.h5ad \
  --model.mode=inversion \
  --model.backbone=mlp \
  --model.latent_dim=256 \
  --training.sample_steps=50 \
  --training.cfg_scale=2.0
```

## 问题汇总
### 问题 1：AE 重建质量差

**解决方案**：
- 增加 `model.latent_dim` (如 128 → 256)
- 增加 `model.hidden_dim` (如 [512, 256] → [1024, 512])
- 调整损失函数 `training.reconstruction_loss` (尝试 poisson)

### 问题 2：RTF 采样不稳定

**解决方案**：
- 增加 `training.sample_steps` (如 50 → 100)
- 调整 `training.cfg_scale` (如 2.0 → 1.5)
- 检查 AE 是否收敛良好

## 致谢
基于以下开源项目：
- PyTorch Lightning
- scimilarity
- Fire
- OmegaConf
- scanpy