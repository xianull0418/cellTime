#!/bin/bash
################################################################################
# RTF Only Benchmark 脚本
# 用于评估训练好的 RTF Only 模型
################################################################################

set -e

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# 配置区域 - 根据实际情况修改
# ============================================================================

# 模型检查点路径（必须指定）
CKPT_PATH="${1:-output/rtf_only_experiment_tigon_inversion/checkpoints/last.ckpt}"

# 数据路径（可选，默认使用训练时的数据路径）
DATA_PATH="${2:-}"

# 输出目录
OUTPUT_DIR="${3:-benchmarks/results/rtf_only_tigon_inversion}"

# 评估参数
BATCH_SIZE=100
SAMPLE_STEPS=20
VIS_CELLS_PER_TIME=50
DEVICE="cuda"
SEED=42

# ============================================================================
# 帮助信息
# ============================================================================

show_help() {
    echo "Usage: $0 [CKPT_PATH] [DATA_PATH] [OUTPUT_DIR]"
    echo ""
    echo "Arguments:"
    echo "  CKPT_PATH   模型检查点路径 (默认: output/rtf_only_experiment_cellrank/checkpoints/last.ckpt)"
    echo "  DATA_PATH   数据路径 (可选，默认使用训练时的配置)"
    echo "  OUTPUT_DIR  输出目录 (默认: benchmarks/results/rtf_only)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 output/my_experiment/checkpoints/best.ckpt"
    echo "  $0 output/my_experiment/checkpoints/best.ckpt /path/to/data.h5ad"
    echo "  $0 output/my_experiment/checkpoints/best.ckpt /path/to/data.h5ad benchmarks/my_results"
    echo ""
    echo "Environment Variables:"
    echo "  CUDA_VISIBLE_DEVICES  指定使用的 GPU (默认: 0)"
    echo "  BATCH_SIZE            批次大小 (默认: 100)"
    echo "  SAMPLE_STEPS          采样步数 (默认: 20)"
    echo "  VIS_CELLS_PER_TIME    每个时间点可视化的细胞数 (默认: 50)"
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# ============================================================================
# 检查文件存在
# ============================================================================

if [[ ! -f "$CKPT_PATH" ]]; then
    echo "Error: 检查点文件不存在: $CKPT_PATH"
    echo ""
    echo "请指定正确的检查点路径，或查看可用的检查点："
    echo "  find output -name '*.ckpt' | head -20"
    exit 1
fi

# ============================================================================
# 设置环境
# ============================================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "========================================"
echo "RTF Only Benchmark"
echo "========================================"
echo "  检查点: $CKPT_PATH"
echo "  数据路径: ${DATA_PATH:-'(使用训练配置)'}"
echo "  输出目录: $OUTPUT_DIR"
echo "  GPU: $CUDA_VISIBLE_DEVICES"
echo "  批次大小: $BATCH_SIZE"
echo "  采样步数: $SAMPLE_STEPS"
echo "  每时间点细胞数: $VIS_CELLS_PER_TIME"
echo "========================================"
echo ""

# ============================================================================
# 创建输出目录
# ============================================================================

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# 运行 Benchmark
# ============================================================================

# 构建命令
CMD="python benchmarks/benchmark_rtf_only.py \
    --ckpt_path=\"$CKPT_PATH\" \
    --output_dir=\"$OUTPUT_DIR\" \
    --batch_size=$BATCH_SIZE \
    --sample_steps=$SAMPLE_STEPS \
    --vis_cells_per_time=$VIS_CELLS_PER_TIME \
    --device=$DEVICE \
    --seed=$SEED"

# 如果指定了数据路径，添加到命令
if [[ -n "$DATA_PATH" ]]; then
    CMD="$CMD --data_path=\"$DATA_PATH\""
fi

echo "Running: $CMD"
echo ""

eval $CMD

# ============================================================================
# 完成
# ============================================================================

echo ""
echo "========================================"
echo "Benchmark 完成!"
echo "========================================"
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "生成的文件:"
ls -la "$OUTPUT_DIR"
echo ""
echo "查看结果:"
echo "  - metrics.csv: 相关性和MSE指标"
echo "  - correlation_dist.png: 相关性分布"
echo "  - trajectory_pca_2d.png: 2D PCA轨迹图"
echo "  - trajectory_per_timepoint.png: 分时间点对比"
echo "  - trajectory_correlation_summary.png: 相关性统计"
echo "  - trajectory_vector_field.png: 向量场可视化"
echo "========================================"
