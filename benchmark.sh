#!/bin/bash
# cellTime Benchmark (Simplified)
# 仅包含核心评估功能

set -e

# 环境设置
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ================= 配置 =================
# 数据路径
TEST_DATA="/gpfs/hybrid/data/public/TEDD/link_cells/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"

# 模型路径
AE_CKPT="output/ae/checkpoints/last.ckpt"
RTF_CKPT="output/rtf_direct_dit/checkpoints/last.ckpt"

# 输出
OUT_DIR="output/benchmark"

# 参数
BATCH=128
STEPS=50
SCALE=2.0
# ========================================

show_help() {
    echo "Usage: bash benchmark.sh [mode]"
    echo "Modes:"
    echo "  full  (Default) Evaluate both AE and RTF"
    echo "  ae    Evaluate AE only"
    echo "  rtf   Evaluate RTF only"
    echo ""
    echo "Env Vars: SAMPLE_STEPS (default 50), BATCH_SIZE (default 128)"
}

# 参数解析
MODE=${1:-full}
if [[ "$MODE" == "help" || "$MODE" == "-h" ]]; then
    show_help
    exit 0
fi

# 允许环境变量覆盖
STEPS=${SAMPLE_STEPS:-$STEPS}
BATCH=${BATCH_SIZE:-$BATCH}

echo "=== Running Benchmark: $MODE ==="
echo "Data: $TEST_DATA"
echo "AE:   $AE_CKPT"
if [[ "$MODE" != "ae" ]]; then
    echo "RTF:  $RTF_CKPT"
fi
echo "Steps: $STEPS | Batch: $BATCH"
echo ""

# 运行 Python 脚本
python utils/benchmark.py \
    --mode="$MODE" \
    --ae_checkpoint="$AE_CKPT" \
    --rtf_checkpoint="$RTF_CKPT" \
    --input_data="$TEST_DATA" \
    --output_dir="$OUT_DIR" \
    --batch_size=$BATCH \
    --sample_steps=$STEPS \
    --cfg_scale=$SCALE

echo ""
echo "Results saved to: $OUT_DIR"
echo "  - metrics.json"
echo "  - ae_eval.png (if AE ran)"
echo "  - rtf_squidiff_eval.png (if RTF ran)"
