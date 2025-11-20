#!/bin/bash
################################################################################
# cellTime Benchmark 评估脚本
# 用途：评估 AE 重建质量和 RTF 时序预测质量
#
# 使用方法：
#   bash benchmark.sh full        # 完整评估（AE + RTF）
#   bash benchmark.sh ae          # 仅评估 AE
#   bash benchmark.sh rtf         # 仅评估 RTF
#   bash benchmark.sh help        # 显示帮助信息
################################################################################

set -e  # 遇到错误立即退出

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置 PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "✓ 项目根目录: $SCRIPT_DIR"
echo "✓ PYTHONPATH: $PYTHONPATH"
echo ""

# ============================================================================
# 配置区域
# ============================================================================

# 测试数据路径（需要包含 next_cell_id 列）
TEST_DATA_PATH="/gpfs/hybrid/data/public/TEDD/link_cells/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"

# 模型 checkpoint 路径
AE_CHECKPOINT="output/ae/checkpoints/last.ckpt"
RTF_CHECKPOINT="output/rtf_direct_dit/checkpoints/last.ckpt"

# 输出目录
OUTPUT_DIR="output/benchmark"

# 评估参数
BATCH_SIZE=64
SAMPLE_STEPS=50
CFG_SCALE=2.0
DEVICE="auto"  # auto/cuda/cpu

# ============================================================================
# 函数定义
# ============================================================================

show_help() {
    echo "========================================"
    echo "cellTime Benchmark 评估脚本"
    echo "========================================"
    echo ""
    echo "使用方法："
    echo "  bash benchmark.sh <mode>"
    echo ""
    echo "评估模式："
    echo "  full   - 完整评估（AE 重建 + RTF 时序预测）"
    echo "  ae     - 仅评估 AE 重建质量"
    echo "  rtf    - 仅评估 RTF 时序预测质量"
    echo "  help   - 显示此帮助信息"
    echo ""
    echo "评估指标："
    echo ""
    echo "  【AE 重建质量】"
    echo "    - MSE/RMSE/MAE: 重建误差"
    echo "    - 基因级别相关性: 每个基因的重建质量"
    echo "    - 细胞级别相关性: 每个细胞的重建质量"
    echo "    - 高变基因重建质量"
    echo "    - 潜空间统计"
    echo ""
    echo "  【RTF 时序预测质量】"
    echo "    - MSE/RMSE/MAE: 预测误差（vs 真实 next_cell）"
    echo "    - 基因级别预测相关性"
    echo "    - 细胞级别预测相关性"
    echo "    - Cosine 相似度"
    echo "    - 时间差异统计"
    echo ""
    echo "环境变量（可选）："
    echo "  SAMPLE_STEPS  - RTF 采样步数（默认 50）"
    echo "  CFG_SCALE     - CFG 强度（默认 2.0）"
    echo "  BATCH_SIZE    - 批次大小（默认 64）"
    echo ""
    echo "示例："
    echo "  # 完整评估"
    echo "  bash benchmark.sh full"
    echo ""
    echo "  # 仅评估 AE"
    echo "  bash benchmark.sh ae"
    echo ""
    echo "  # 自定义参数"
    echo "  SAMPLE_STEPS=100 CFG_SCALE=3.0 bash benchmark.sh full"
    echo ""
    echo "当前配置："
    echo "  测试数据: $TEST_DATA_PATH"
    echo "  AE 模型: $AE_CHECKPOINT"
    echo "  RTF 模型: $RTF_CHECKPOINT"
    echo "  输出目录: $OUTPUT_DIR"
    echo "========================================"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ 错误: 文件不存在 - $1"
        exit 1
    fi
}

run_full_benchmark() {
    echo "========================================"
    echo "运行完整评估（AE + RTF）"
    echo "========================================"
    
    # 检查必要文件
    check_file "$TEST_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    check_file "$RTF_CHECKPOINT"
    
    # 参数设置
    local sample_steps=${SAMPLE_STEPS:-$SAMPLE_STEPS}
    local cfg_scale=${CFG_SCALE:-$CFG_SCALE}
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    echo ""
    echo "配置参数："
    echo "  测试数据: $TEST_DATA_PATH"
    echo "  采样步数: $sample_steps"
    echo "  CFG 强度: $cfg_scale"
    echo "  批次大小: $batch_size"
    echo "  输出目录: $OUTPUT_DIR"
    echo ""
    
    # 运行评估
    python utils/benchmark.py full \
        --ae_checkpoint="$AE_CHECKPOINT" \
        --rtf_checkpoint="$RTF_CHECKPOINT" \
        --input_data="$TEST_DATA_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --batch_size=$batch_size \
        --sample_steps=$sample_steps \
        --cfg_scale=$cfg_scale \
        --device="$DEVICE"
    
    echo ""
    echo "✓ 完整评估完成！"
    echo ""
    echo "查看结果："
    echo "  - 数值结果: $OUTPUT_DIR/benchmark_results.json"
    echo "  - AE 可视化: $OUTPUT_DIR/ae_correlation_distributions.png"
    echo "  - RTF 可视化: $OUTPUT_DIR/rtf_prediction_distributions.png"
}

run_ae_benchmark() {
    echo "========================================"
    echo "运行 AE 重建质量评估"
    echo "========================================"
    
    # 检查必要文件
    check_file "$TEST_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    
    # 参数设置
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    echo ""
    echo "配置参数："
    echo "  测试数据: $TEST_DATA_PATH"
    echo "  批次大小: $batch_size"
    echo "  输出目录: $OUTPUT_DIR"
    echo ""
    
    # 运行评估
    python utils/benchmark.py ae \
        --ae_checkpoint="$AE_CHECKPOINT" \
        --input_data="$TEST_DATA_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --batch_size=$batch_size \
        --device="$DEVICE"
    
    echo ""
    echo "✓ AE 评估完成！"
    echo ""
    echo "查看结果："
    echo "  - 数值结果: $OUTPUT_DIR/ae_benchmark_results.json"
    echo "  - 可视化: $OUTPUT_DIR/ae_correlation_distributions.png"
}

run_rtf_benchmark() {
    echo "========================================"
    echo "运行 RTF 时序预测质量评估"
    echo "========================================"
    
    # 检查必要文件
    check_file "$TEST_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    check_file "$RTF_CHECKPOINT"
    
    # 参数设置
    local sample_steps=${SAMPLE_STEPS:-$SAMPLE_STEPS}
    local cfg_scale=${CFG_SCALE:-$CFG_SCALE}
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    echo ""
    echo "配置参数："
    echo "  测试数据: $TEST_DATA_PATH"
    echo "  采样步数: $sample_steps"
    echo "  CFG 强度: $cfg_scale"
    echo "  批次大小: $batch_size"
    echo "  输出目录: $OUTPUT_DIR"
    echo ""
    
    # 运行评估
    python utils/benchmark.py rtf \
        --ae_checkpoint="$AE_CHECKPOINT" \
        --rtf_checkpoint="$RTF_CHECKPOINT" \
        --input_data="$TEST_DATA_PATH" \
        --output_dir="$OUTPUT_DIR" \
        --batch_size=$batch_size \
        --sample_steps=$sample_steps \
        --cfg_scale=$cfg_scale \
        --device="$DEVICE"
    
    echo ""
    echo "✓ RTF 评估完成！"
    echo ""
    echo "查看结果："
    echo "  - 数值结果: $OUTPUT_DIR/rtf_benchmark_results.json"
    echo "  - 可视化: $OUTPUT_DIR/rtf_prediction_distributions.png"
}

# ============================================================================
# 主程序
# ============================================================================

if [ $# -eq 0 ]; then
    echo "❌ 错误: 缺少评估模式参数"
    echo ""
    show_help
    exit 1
fi

MODE=$1

case $MODE in
    full)
        run_full_benchmark
        ;;
    ae)
        run_ae_benchmark
        ;;
    rtf)
        run_rtf_benchmark
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ 错误: 未知的评估模式 '$MODE'"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Benchmark 评估完成！"
echo "========================================"

