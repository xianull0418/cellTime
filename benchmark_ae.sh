#!/bin/bash
# Benchmark Script for Autoencoder
# Usage: ./benchmark_ae.sh [checkpoint_path] [data_dir]

set -e

# Default paths
CKPT_PATH=${1:-"output/ae_test_10epochs/checkpoints/last.ckpt"}
DATA_DIR=${2:-"data/ae_test_subset"}

# Environment setup
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Autoencoder Benchmark"
echo "=========================================="
echo ""
echo "Checkpoint: $CKPT_PATH"
echo "Data Dir:   $DATA_DIR"
echo ""

# Check if checkpoint exists
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please run test_train_ae.sh first to create the test data subset."
    exit 1
fi

# Run benchmark
python benchmarks/benchmark_ae.py \
    --ckpt_path="$CKPT_PATH" \
    --data_dir="$DATA_DIR" \
    --batch_size=2048 \
    --max_batches=200 \
    --device="cuda"

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
