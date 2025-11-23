#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0


DATA_PATH="/gpfs/hybrid/data/public/PerturBase/drug_perturb.true_time/test/5.link_cells/GSE134839.Erlotinib.link.h5ad"
TEMPORAL_DATA_PATH="/gpfs/hybrid/data/public/PerturBase/drug_perturb.true_time/test/5.link_cells/GSE134839.Erlotinib.link.h5ad"

# ==========================================
# 配置区域
# ==========================================
# AE 训练模式选择: 
#   "scimilarity" - 加载预训练权重微调 (Latent=128)
#   "scratch"     - 从头开始训练 (Latent=256)
#   "pretrained"  - 直接使用预训练权重 (Latent=128, 需基因映射)
AE_MODE="pretrained" 

if [ "$AE_MODE" = "scimilarity" ]; then
    echo ">>> 使用模式: scimilarity (微调预训练模型)"
    AE_OUTPUT_DIR="output/ae_finetune"
    LATENT_DIM=128
    AE_CMD="python train_ae.py \
      --config_path=config/ae_scimilarity.yaml \
      --data_path=${DATA_PATH} \
      --logging__output_dir=${AE_OUTPUT_DIR}"
    RTF_EXTRA_ARGS=""
elif [ "$AE_MODE" = "pretrained" ]; then
    echo ">>> 使用模式: pretrained (直接使用预训练权重，数据映射)"
    AE_OUTPUT_DIR="output/ae_pretrained"
    LATENT_DIM=128
    # 关键配置：
    # 1. target_genes_path: 指定 scimilarity 的基因列表
    # 2. reset_input_output_layers=false: 保持原有输入输出层
    # 3. max_epochs=1: 仅运行极少步数以生成checkpoint（相当于不做训练）
    # 4. learning_rate=0: 确保权重不更新
    # 5. n_genes=28231: 显式指定基因数，与 gene_order.tsv 一致
    AE_CMD="python train_ae.py \
      --config_path=config/ae_scimilarity.yaml \
      --data_path=${DATA_PATH} \
      --data__target_genes_path=data/models/scimilarity/gene_order.tsv \
      --model__n_genes=28231 \
      --model__reset_input_output_layers=false \
      --model__freeze_encoder_layers=true \
      --training__max_epochs=1 \
      --training__learning_rate=0.0 \
      --logging__output_dir=${AE_OUTPUT_DIR}"
    
    # RTF 也需要知道目标基因列表，以便进行相同的数据映射
    RTF_EXTRA_ARGS="--data__target_genes_path=data/models/scimilarity/gene_order.tsv"
else
    echo ">>> 使用模式: scratch (从头训练)"
    AE_OUTPUT_DIR="output/ae"
    LATENT_DIM=256
    AE_CMD="python train_ae.py \
      --data_path=${DATA_PATH} \
      --model__n_genes=3000 \
      --model__latent_dim=${LATENT_DIM} \
      --model__hidden_dim='[1024,512]' \
      --training__batch_size=256 \
      --training__learning_rate=1e-3 \
      --training__max_epochs=100 \
      --logging__output_dir=${AE_OUTPUT_DIR}"
      
    RTF_EXTRA_ARGS=""
fi

# ==========================================
# 阶段 1：训练 Autoencoder (或加载预训练)
# ==========================================

echo ""
echo "[阶段 1] 准备 Autoencoder..."
echo "执行命令: $AE_CMD"

# 执行训练
eval $AE_CMD

echo ""
echo "AE 准备完成！"
echo "Checkpoint: ${AE_OUTPUT_DIR}/checkpoints/last.ckpt"
echo "Latent Dim: ${LATENT_DIM}"
echo ""

# ==========================================
# 阶段 2a：训练 RTF (Direct 模式 + DiT)
# ==========================================
echo ""
echo "[阶段 2a] 训练 RTF - Direct 模式 + DiT 骨干网络..."

# Inversion 模式需要启用条件信息来区分不同的映射方向
# 注意：如果 AE_MODE=pretrained，数据会被自动映射到 28231 维
python train_rtf.py \
  --ae_checkpoint=${AE_OUTPUT_DIR}/checkpoints/last.ckpt \
  --data_path=${TEMPORAL_DATA_PATH} \
  --model__mode=direct \
  --model__backbone=unet \
  --model__latent_dim=${LATENT_DIM} \
  --training__batch_size=128 \
  --training__learning_rate=1e-4 \
  --training__max_epochs=10 \
  --training__sample_steps=50 \
  --training__cfg_scale=2.0 \
  --logging__output_dir=output/rtf_direct_unet_${AE_MODE}_1123_debug \
  $RTF_EXTRA_ARGS

echo ""
echo "RTF (Direct+DiT) 训练完成！"
echo ""

echo "启动 TensorBoard 查看训练日志："
echo "  tensorboard --logdir=output/"
