#!/bin/bash
################################################################################
# cellTime 推理脚本
# 用途：使用训练好的 AE + RTF 模型进行单细胞时序预测
# 
# 使用方法：
#   bash infer.sh predict          # 单次时间点预测
#   bash infer.sh trajectory       # 多时间点轨迹预测
#   bash infer.sh encode           # 仅编码到潜空间
#   bash infer.sh help             # 显示帮助信息
################################################################################

set -e  # 遇到错误立即退出

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 设置 PYTHONPATH 以确保能导入项目模块
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "✓ 项目根目录: $SCRIPT_DIR"
echo "✓ PYTHONPATH: $PYTHONPATH"
echo ""

# ============================================================================
# 配置区域 - 根据实际情况修改
# ============================================================================

# 数据路径
TEMPORAL_DATA_PATH="/gpfs/hybrid/data/public/TEDD/link_cells/hs_AD_Brain_Cerebellum[Organoid]_36-a.link.h5ad"

# 模型 checkpoint 路径（
AE_CHECKPOINT="output/ae/checkpoints/last.ckpt"
RTF_CHECKPOINT="output/rtf_direct_dit/checkpoints/last.ckpt"

# 输出目录
OUTPUT_DIR="output/inference_results"

# 推理参数
BATCH_SIZE=64
SAMPLE_STEPS=50
CFG_SCALE=2.0
DEVICE="auto"  # auto/cuda/cpu

# ============================================================================
# 函数定义
# ============================================================================

# 显示帮助信息
show_help() {
    echo "========================================"
    echo "cellTime 推理脚本"
    echo "========================================"
    echo ""
    echo "使用方法："
    echo "  bash infer.sh <mode> [options]"
    echo ""
    echo "推理模式："
    echo "  predict      - 单次时间点预测（默认 t=0 -> t=1）"
    echo "  trajectory   - 多时间点轨迹预测"
    echo "  encode       - 仅编码到潜空间（不需要 RTF 模型）"
    echo "  help         - 显示此帮助信息"
    echo ""
    echo "环境变量（可选）："
    echo "  TARGET_TIME    - 目标时间点（默认 1.0）"
    echo "  START_TIME     - 起始时间点（默认 0.0）"
    echo "  TIME_POINTS    - 轨迹时间点列表（默认 '[0.0,0.5,1.0,1.5,2.0]'）"
    echo "  SAMPLE_STEPS   - 采样步数（默认 50）"
    echo "  CFG_SCALE      - CFG 强度（默认 2.0）"
    echo "  BATCH_SIZE     - 批次大小（默认 64）"
    echo ""
    echo "示例："
    echo "  # 默认参数预测"
    echo "  bash infer.sh predict"
    echo ""
    echo "  # 自定义目标时间"
    echo "  TARGET_TIME=2.0 bash infer.sh predict"
    echo ""
    echo "  # 多时间点轨迹"
    echo "  bash infer.sh trajectory"
    echo ""
    echo "  # 仅编码"
    echo "  bash infer.sh encode"
    echo ""
    echo "当前配置："
    echo "  数据路径: $TEMPORAL_DATA_PATH"
    echo "  AE 模型: $AE_CHECKPOINT"
    echo "  RTF 模型: $RTF_CHECKPOINT"
    echo "  输出目录: $OUTPUT_DIR"
    echo "========================================"
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ 错误: 文件不存在 - $1"
        exit 1
    fi
}

# 创建输出目录
create_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    echo "✓ 输出目录: $OUTPUT_DIR"
}

# 单次时间点预测
run_predict() {
    echo "========================================"
    echo "运行单次时间点预测"
    echo "========================================"
    
    # 检查必要文件
    check_file "$TEMPORAL_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    check_file "$RTF_CHECKPOINT"
    
    # 创建输出目录
    create_output_dir
    
    # 参数设置（支持环境变量覆盖）
    local start_time=${START_TIME:-0.0}
    local target_time=${TARGET_TIME:-1.0}
    local sample_steps=${SAMPLE_STEPS:-$SAMPLE_STEPS}
    local cfg_scale=${CFG_SCALE:-$CFG_SCALE}
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    # 输出文件名
    local output_file="${OUTPUT_DIR}/predict_t${start_time}_to_t${target_time}.h5ad"
    
    echo ""
    echo "配置参数："
    echo "  起始时间: $start_time"
    echo "  目标时间: $target_time"
    echo "  采样步数: $sample_steps"
    echo "  CFG 强度: $cfg_scale"
    echo "  批次大小: $batch_size"
    echo "  输出文件: $output_file"
    echo ""
    
    # 运行推理
    python utils/inference.py predict \
        --ae_checkpoint="$AE_CHECKPOINT" \
        --rtf_checkpoint="$RTF_CHECKPOINT" \
        --input_data="$TEMPORAL_DATA_PATH" \
        --output_path="$output_file" \
        --start_time=$start_time \
        --target_time=$target_time \
        --sample_steps=$sample_steps \
        --cfg_scale=$cfg_scale \
        --batch_size=$batch_size \
        --device="$DEVICE"
    
    echo ""
    echo "✓ 预测完成！结果保存在: $output_file"
}

# 多时间点轨迹预测
run_trajectory() {
    echo "========================================"
    echo "运行多时间点轨迹预测"
    echo "========================================"
    
    # 检查必要文件
    check_file "$TEMPORAL_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    check_file "$RTF_CHECKPOINT"
    
    # 创建输出目录
    create_output_dir
    
    # 参数设置（支持环境变量覆盖）
    local time_points=${TIME_POINTS:-"[0.0,0.5,1.0,1.5,2.0]"}
    local sample_steps=${SAMPLE_STEPS:-$SAMPLE_STEPS}
    local cfg_scale=${CFG_SCALE:-$CFG_SCALE}
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    # 输出文件名
    local output_file="${OUTPUT_DIR}/trajectory.h5ad"
    
    echo ""
    echo "配置参数："
    echo "  时间点序列: $time_points"
    echo "  采样步数: $sample_steps"
    echo "  CFG 强度: $cfg_scale"
    echo "  批次大小: $batch_size"
    echo "  输出文件: $output_file"
    echo ""
    
    # 运行推理
    python utils/inference.py predict_trajectory \
        --ae_checkpoint="$AE_CHECKPOINT" \
        --rtf_checkpoint="$RTF_CHECKPOINT" \
        --input_data="$TEMPORAL_DATA_PATH" \
        --output_path="$output_file" \
        --time_points="$time_points" \
        --sample_steps=$sample_steps \
        --cfg_scale=$cfg_scale \
        --batch_size=$batch_size \
        --device="$DEVICE"
    
    echo ""
    echo "✓ 轨迹预测完成！结果保存在: $output_file"
}

# 仅编码到潜空间
run_encode() {
    echo "========================================"
    echo "编码数据到潜空间"
    echo "========================================"
    
    # 检查必要文件
    check_file "$TEMPORAL_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    
    # 创建输出目录
    create_output_dir
    
    # 参数设置
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    # 输出文件名
    local output_file="${OUTPUT_DIR}/latent_encoded.h5ad"
    
    echo ""
    echo "配置参数："
    echo "  批次大小: $batch_size"
    echo "  输出文件: $output_file"
    echo ""
    
    # 运行编码
    python utils/inference.py encode_data \
        --ae_checkpoint="$AE_CHECKPOINT" \
        --input_data="$TEMPORAL_DATA_PATH" \
        --output_path="$output_file" \
        --batch_size=$batch_size \
        --device="$DEVICE"
    
    echo ""
    echo "✓ 编码完成！结果保存在: $output_file"
}

# ============================================================================
# 主程序
# ============================================================================

# 检查是否有参数
if [ $# -eq 0 ]; then
    echo "❌ 错误: 缺少推理模式参数"
    echo ""
    show_help
    exit 1
fi

# 解析命令
MODE=$1

case $MODE in
    predict)
        run_predict
        ;;
    trajectory)
        run_trajectory
        ;;
    encode)
        run_encode
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ 错误: 未知的推理模式 '$MODE'"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "任务完成！"
echo "========================================"

