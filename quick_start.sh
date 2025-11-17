#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0


DATA_PATH="/gpfs/hybrid/data/public/TEDD/link_cells/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"
TEMPORAL_DATA_PATH="/gpfs/hybrid/data/public/TEDD/link_cells/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"

# 阶段 1：训练 Autoencoder

echo ""
echo "[阶段 1] 训练 Autoencoder..."

python train_ae.py \
  --data_path=${DATA_PATH} \
  --model__n_genes=3000 \
  --model__latent_dim=256 \
  --model__hidden_dim='[1024,512]' \
  --training__batch_size=256 \
  --training__learning_rate=1e-3 \
  --training__max_epochs=100 \
  --logging__output_dir=output/ae

echo ""
echo "AE 训练完成！Checkpoint: output/ae/checkpoints/last.ckpt"
echo ""

# 阶段 2a：训练 RTF (Direct 模式 + DiT)
echo ""
echo "[阶段 2a] 训练 RTF - Direct 模式 + DiT 骨干网络..."

python train_rtf.py \
  --ae_checkpoint=output/ae/checkpoints/last.ckpt \
  --data_path=${TEMPORAL_DATA_PATH} \
  --model__mode=direct \
  --model__backbone=dit \
  --model__latent_dim=256 \
  --training__batch_size=128 \
  --training__learning_rate=1e-4 \
  --training__max_epochs=100 \
  --training__sample_steps=50 \
  --training__cfg_scale=2.0 \
  --logging__output_dir=output/rtf_direct_dit

echo ""
echo "RTF (Direct+DiT) 训练完成！"
echo ""

echo "启动 TensorBoard 查看训练日志："
echo "  tensorboard --logdir=output/"

