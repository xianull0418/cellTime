#!/bin/bash
################################################################################
# cellTime RTF Only Inference Script
# Usage: Use trained RTF Only models for single-cell temporal prediction
################################################################################

set -e

# Get project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ============================================================================
# Configuration
# ============================================================================

# Data Path
INPUT_DATA="/gpfs/hybrid/data/public/PerturBase/drug_perturb.true_time/test/5.link_cells/GSE134839.Erlotinib.link.h5ad"

# Checkpoint Path (Update this after training)
CHECKPOINT="output/rtf_only_experiment/checkpoints/last-v3.ckpt"

# Gene Order File (Crucial for alignment)
GENE_ORDER="gene_order.tsv"

# Output Directory
OUTPUT_DIR="output/inference_rtf_only"

# Parameters
BATCH_SIZE=64
SAMPLE_STEPS=50
CFG_SCALE=2.0
DEVICE="auto"

# ============================================================================
# Functions
# ============================================================================

run_predict() {
    python utils/inference_rtf_only.py predict \
        --checkpoint="$CHECKPOINT" \
        --input_data="$INPUT_DATA" \
        --output_path="$OUTPUT_DIR/prediction.h5ad" \
        --target_genes_path="$GENE_ORDER" \
        --sample_steps=$SAMPLE_STEPS \
        --cfg_scale=$CFG_SCALE \
        --batch_size=$BATCH_SIZE \
        --device="$DEVICE"
}

run_trajectory() {
    python utils/inference_rtf_only.py predict_trajectory \
        --checkpoint="$CHECKPOINT" \
        --input_data="$INPUT_DATA" \
        --output_path="$OUTPUT_DIR/trajectory.h5ad" \
        --target_genes_path="$GENE_ORDER" \
        --sample_steps=$SAMPLE_STEPS \
        --cfg_scale=$CFG_SCALE \
        --batch_size=$BATCH_SIZE \
        --device="$DEVICE"
}

run_visualize() {
    local viz_file=${VIZ_FILE:-"${OUTPUT_DIR}/trajectory.h5ad"}
    local output_img="${viz_file%.*}.png"
    
    echo "Visualizing $viz_file..."
    
    python utils/inference.py visualize \
        --data_path="$viz_file" \
        --output_path="$output_img" \
        --basis="umap"
}

# ============================================================================
# Main
# ============================================================================

MODE=$1

if [ "$MODE" == "predict" ]; then
    run_predict
elif [ "$MODE" == "trajectory" ]; then
    run_trajectory
elif [ "$MODE" == "visualize" ]; then
    run_visualize
else
    echo "Usage: bash infer_rtf_only.sh [predict|trajectory|visualize]"
    exit 1
fi

