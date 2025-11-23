#!/bin/bash
################################################################################
# RTF Only 训练脚本 (不使用 Autoencoder)
# 直接在原始基因表达空间训练 Rectified Flow
################################################################################

set -e  # 遇到错误立即退出

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "✓ 项目根目录: $SCRIPT_DIR"
echo ""

# ============================================================================
# 配置区域 - 根据实际情况修改
# ============================================================================

# 数据路径
DATA_PATH="/gpfs/hybrid/data/public/PerturBase/drug_perturb.true_time/test/5.link_cells/GSE134839.Erlotinib.link.h5ad"

# 输出目录
OUTPUT_DIR="output/rtf_only_experiment"

# 骨干网络选择: "mlp", "dit", "unet"
# 注意: UNet 要求基因数能被 2^(depth-1) 整除。对于任意基因数，推荐使用 MLP 或 DiT。
BACKBONE="dit" 

# 训练参数
BATCH_SIZE=128
MAX_EPOCHS=100
LEARNING_RATE=1e-4

# ============================================================================
# 运行训练
# ============================================================================

echo "========================================"
echo "开始训练 RTF Only 模型"
echo "========================================"
echo "  数据路径: $DATA_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  骨干网络: $BACKBONE"
echo "========================================"

# 执行训练命令
python train_rtf_only.py \
  --config_path=config/rtf_only.yaml \
  --data_path="$DATA_PATH" \
  --model__backbone="$BACKBONE" \
  --training__batch_size=$BATCH_SIZE \
  --training__max_epochs=$MAX_EPOCHS \
  --training__learning_rate=$LEARNING_RATE \
  --logging__output_dir="$OUTPUT_DIR"

echo ""
echo "========================================"
echo "训练完成！"
echo "查看日志: tensorboard --logdir=$OUTPUT_DIR"
echo "========================================"

