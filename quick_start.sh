#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0


DATA_PATH="/gpfs/hybrid/data/public/TEDD/link_cells/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"
TEMPORAL_DATA_PATH="/gpfs/hybrid/data/public/TEDD/link_cells/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"

# ==========================================
# 配置区域
# ==========================================
# AE 训练模式选择: 
#   "scimilarity" - 加载预训练权重微调 (Latent=128)
#   "scratch"     - 从头开始训练 (Latent=256)
AE_MODE="scimilarity" 

if [ "$AE_MODE" = "scimilarity" ]; then
    echo ">>> 使用模式: scimilarity (微调预训练模型)"
    AE_OUTPUT_DIR="output/ae_finetune"
    LATENT_DIM=128
    # 使用配置文件，并在命令行覆盖数据路径
    AE_CMD="python train_ae.py \
      --config_path=config/ae_scimilarity.yaml \
      --data_path=${DATA_PATH} \
      --logging__output_dir=${AE_OUTPUT_DIR}"
else
    echo ">>> 使用模式: scratch (从头训练)"
    AE_OUTPUT_DIR="output/ae"
    LATENT_DIM=256
    # 完全通过命令行参数配置
    AE_CMD="python train_ae.py \
      --data_path=${DATA_PATH} \
      --model__n_genes=3000 \
      --model__latent_dim=${LATENT_DIM} \
      --model__hidden_dim='[1024,512]' \
      --training__batch_size=256 \
      --training__learning_rate=1e-3 \
      --training__max_epochs=100 \
      --logging__output_dir=${AE_OUTPUT_DIR}"
fi

# ==========================================
# 阶段 1：训练 Autoencoder
# ==========================================

echo ""
echo "[阶段 1] 训练 Autoencoder..."
echo "执行命令: $AE_CMD"

# 执行训练
eval $AE_CMD

echo ""
echo "AE 训练完成！"
echo "Checkpoint: ${AE_OUTPUT_DIR}/checkpoints/last.ckpt"
echo "Latent Dim: ${LATENT_DIM}"
echo ""

# ==========================================
# 阶段 2a：训练 RTF (Direct 模式 + DiT)
# ==========================================
echo ""
echo "[阶段 2a] 训练 RTF - Direct 模式 + DiT 骨干网络..."

python train_rtf.py \
  --ae_checkpoint=${AE_OUTPUT_DIR}/checkpoints/last.ckpt \
  --data_path=${TEMPORAL_DATA_PATH} \
  --model__mode=direct \
  --model__backbone=dit \
  --model__latent_dim=${LATENT_DIM} \
  --training__batch_size=128 \
  --training__learning_rate=1e-4 \
  --training__max_epochs=100 \
  --training__sample_steps=50 \
  --training__cfg_scale=2.0 \
  --logging__output_dir=output/rtf_direct_dit_${AE_MODE}_debug_1120

echo ""
echo "RTF (Direct+DiT) 训练完成！"
echo ""

echo "启动 TensorBoard 查看训练日志："
echo "  tensorboard --logdir=output/"
