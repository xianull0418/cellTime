#!/bin/bash
################################################################################
# cellTime Inference Script
# Usage: Use trained AE + RTF models for single-cell temporal prediction
# 
# Usage:
#   bash infer.sh predict          # Single time point prediction
#   bash infer.sh trajectory       # Multi-time point trajectory prediction
#   bash infer.sh encode           # Encode to latent space only
#   bash infer.sh visualize        # Visualize results (UMAP/PCA)
#   bash infer.sh help             # Show help
################################################################################

set -e  # Exit immediately on error

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "✓ Project Root: $SCRIPT_DIR"
echo "✓ PYTHONPATH: $PYTHONPATH"
echo ""

# ============================================================================
# Configuration - Update as needed
# ============================================================================

# Data Path
TEMPORAL_DATA_PATH="/gpfs/hybrid/data/public/PerturBase/drug_perturb.true_time/test/5.link_cells/GSE134839.Erlotinib.link.h5ad"

# Model Checkpoint Paths (Updated to current best models)
AE_CHECKPOINT="output/ae_pretrained/checkpoints/last.ckpt"
RTF_CHECKPOINT="output/rtf_direct_unet_pretrained_1123_debug/checkpoints/last.ckpt"

# Output Directory
OUTPUT_DIR="output/inference_results"

# Inference Parameters
BATCH_SIZE=64
SAMPLE_STEPS=50
CFG_SCALE=2.0
DEVICE="auto"  # auto/cuda/cpu

# ============================================================================
# Function Definitions
# ============================================================================

# Show Help
show_help() {
    echo "========================================"
    echo "cellTime Inference Script"
    echo "========================================"
    echo ""
    echo "Usage:"
    echo "  bash infer.sh <mode> [options]"
    echo ""
    echo "Modes:"
    echo "  predict      - Single time point prediction (default t=0 -> t=1)"
    echo "  trajectory   - Multi-time point trajectory prediction"
    echo "  encode       - Encode to latent space only"
    echo "  visualize    - Visualize results (UMAP/PCA)"
    echo "  help         - Show this help message"
    echo ""
    echo "Environment Variables (Optional):"
    echo "  TARGET_TIME    - Target time point (default 1.0)"
    echo "  START_TIME     - Start time point (default 0.0)"
    echo "  TIME_POINTS    - Trajectory time points (default '[0.0,0.5,1.0,1.5,2.0]')"
    echo "  SAMPLE_STEPS   - Sampling steps (default 50)"
    echo "  CFG_SCALE      - CFG Scale (default 2.0)"
    echo "  BATCH_SIZE     - Batch size (default 64)"
    echo "  VIZ_FILE       - File to visualize (for visualize mode)"
    echo ""
    echo "Examples:"
    echo "  # Default prediction"
    echo "  bash infer.sh predict"
    echo ""
    echo "  # Custom target time"
    echo "  TARGET_TIME=2.0 bash infer.sh predict"
    echo ""
    echo "  # Multi-point trajectory"
    echo "  bash infer.sh trajectory"
    echo ""
    echo "  # Visualize results"
    echo "  VIZ_FILE=output/inference_results/trajectory.h5ad bash infer.sh visualize"
    echo ""
    echo "Current Configuration:"
    echo "  Data Path: $TEMPORAL_DATA_PATH"
    echo "  AE Model: $AE_CHECKPOINT"
    echo "  RTF Model: $RTF_CHECKPOINT"
    echo "  Output Dir: $OUTPUT_DIR"
    echo "========================================"
}

# Check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Error: File not found - $1"
        exit 1
    fi
}

# Create output directory
create_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    echo "✓ Output Directory: $OUTPUT_DIR"
}

# Single Time Point Prediction
run_predict() {
    echo "========================================"
    echo "Running Single Time Point Prediction"
    echo "========================================"
    
    check_file "$TEMPORAL_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    check_file "$RTF_CHECKPOINT"
    
    create_output_dir
    
    local start_time=${START_TIME:-0.0}
    local target_time=${TARGET_TIME:-1.0}
    local sample_steps=${SAMPLE_STEPS:-$SAMPLE_STEPS}
    local cfg_scale=${CFG_SCALE:-$CFG_SCALE}
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    local output_file="${OUTPUT_DIR}/predict_t${start_time}_to_t${target_time}.h5ad"
    
    echo ""
    echo "Parameters:"
    echo "  Start Time: $start_time"
    echo "  Target Time: $target_time"
    echo "  Steps: $sample_steps"
    echo "  CFG Scale: $cfg_scale"
    echo "  Batch Size: $batch_size"
    echo "  Output File: $output_file"
    echo ""
    
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
    echo "✓ Prediction completed! Saved to: $output_file"
}

# Trajectory Prediction
run_trajectory() {
    echo "========================================"
    echo "Running Trajectory Prediction"
    echo "========================================"
    
    check_file "$TEMPORAL_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    check_file "$RTF_CHECKPOINT"
    
    create_output_dir
    
    local time_points=${TIME_POINTS:-"[0.0,0.5,1.0,1.5,2.0]"}
    local sample_steps=${SAMPLE_STEPS:-$SAMPLE_STEPS}
    local cfg_scale=${CFG_SCALE:-$CFG_SCALE}
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    local output_file="${OUTPUT_DIR}/trajectory.h5ad"
    
    echo ""
    echo "Parameters:"
    echo "  Time Points: $time_points"
    echo "  Steps: $sample_steps"
    echo "  CFG Scale: $cfg_scale"
    echo "  Batch Size: $batch_size"
    echo "  Output File: $output_file"
    echo ""
    
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
    echo "✓ Trajectory prediction completed! Saved to: $output_file"
}

# Encode to Latent Space
run_encode() {
    echo "========================================"
    echo "Encoding Data to Latent Space"
    echo "========================================"
    
    check_file "$TEMPORAL_DATA_PATH"
    check_file "$AE_CHECKPOINT"
    
    create_output_dir
    
    local batch_size=${BATCH_SIZE:-$BATCH_SIZE}
    
    local output_file="${OUTPUT_DIR}/latent_encoded.h5ad"
    
    echo ""
    echo "Parameters:"
    echo "  Batch Size: $batch_size"
    echo "  Output File: $output_file"
    echo ""
    
    python utils/inference.py encode_data \
        --ae_checkpoint="$AE_CHECKPOINT" \
        --input_data="$TEMPORAL_DATA_PATH" \
        --output_path="$output_file" \
        --batch_size=$batch_size \
        --device="$DEVICE"
    
    echo ""
    echo "✓ Encoding completed! Saved to: $output_file"
}

# Visualization
run_visualize() {
    echo "========================================"
    echo "Visualizing Results"
    echo "========================================"
    
    local viz_file=${VIZ_FILE:-"${OUTPUT_DIR}/trajectory.h5ad"}
    local output_img="${viz_file%.*}.png"
    
    check_file "$viz_file"
    
    echo ""
    echo "Parameters:"
    echo "  Input File: $viz_file"
    echo "  Output Image: $output_img"
    echo ""
    
    python utils/inference.py visualize \
        --data_path="$viz_file" \
        --output_path="$output_img" \
        --basis="umap"
        
    echo ""
    echo "✓ Visualization completed! Saved to: $output_img"
}

# ============================================================================
# Main Execution
# ============================================================================

if [ $# -eq 0 ]; then
    echo "❌ Error: Missing inference mode"
    echo ""
    show_help
    exit 1
fi

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
    visualize)
        run_visualize
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "❌ Error: Unknown mode '$MODE'"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Task Completed!"
echo "========================================"
