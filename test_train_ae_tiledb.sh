#!/bin/bash
# Test TileDB Training Script
# Uses a small subset of TileDB data for quick testing
#
# Usage: ./test_train_ae_tiledb.sh [OUTPUT_DIR]

set -e

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL Settings
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp83s0f0
export NCCL_NVLS_ENABLE=0
export NCCL_TIMEOUT=1800

# Paths
SOURCE_TILEDB_DIR="data/ae_tiledb"
TEST_TILEDB_DIR="data/ae_test_tiledb"
OUTPUT_DIR="${1:-output/ae_test_tiledb}"

echo "=========================================="
echo "Test TileDB Autoencoder Training"
echo "=========================================="
echo ""

# Check if source TileDB exists
if [ ! -d "$SOURCE_TILEDB_DIR/train_tiledb" ]; then
    echo "ERROR: Source TileDB data not found at: $SOURCE_TILEDB_DIR"
    echo ""
    echo "Please run preprocessing first:"
    echo "  ./preprocess_tiledb.sh"
    exit 1
fi

echo "Creating test subset of TileDB data..."
echo ""

# Create test directory (we'll just symlink the TileDB directories)
# TileDB supports slicing, so we can use the same data
rm -rf "$TEST_TILEDB_DIR"
mkdir -p "$TEST_TILEDB_DIR"

# Symlink the TileDB directories
for split in train val test ood; do
    src="$SOURCE_TILEDB_DIR/${split}_tiledb"
    if [ -d "$src" ]; then
        ln -s "$(realpath $src)" "$TEST_TILEDB_DIR/${split}_tiledb"
        echo "  Linked: ${split}_tiledb"
    fi
done

echo ""
echo "Test data ready (using symlinks to full data)"
echo "Note: TileDB lazy loading means only accessed data is read"
echo ""

# Show info
for split in train val; do
    tiledb_dir="$TEST_TILEDB_DIR/${split}_tiledb"
    if [ -d "$tiledb_dir" ] && [ -f "$tiledb_dir/metadata.json" ]; then
        n_cells=$(python3 -c "import json; print(f\"{json.load(open('$tiledb_dir/metadata.json'))['n_cells']:,}\")" 2>/dev/null || echo "?")
        echo "  ${split}: $n_cells cells (will only load what's needed)"
    fi
done
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/logs"

# Log file
TRAIN_LOG="$OUTPUT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Starting Test Training (3 epochs)..."
echo "=========================================="
echo ""
echo "Logs: $TRAIN_LOG"
echo ""

# Train with reduced epochs and batches for testing
python train_ae.py \
    --config_path="config/ae.yaml" \
    --data.dataset_type="tiledb" \
    --data.processed_path.train="$TEST_TILEDB_DIR/train_tiledb" \
    --data.processed_path.val="$TEST_TILEDB_DIR/val_tiledb" \
    --data.processed_path.test="$TEST_TILEDB_DIR/test_tiledb" \
    --data.processed_path.ood="$TEST_TILEDB_DIR/ood_tiledb" \
    --logging.output_dir="$OUTPUT_DIR" \
    --training.max_epochs=3 \
    --training.batch_size=2048 \
    --accelerator.precision="16-mixed" \
    --data.num_workers=0 \
    --debug=true \
    2>&1 | tee "$TRAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Test Training COMPLETE!"
    echo ""

    # Check for val_loss in logs
    if grep -q "val_loss" "$TRAIN_LOG"; then
        echo "Validation metrics found in logs"
        grep "val_loss" "$TRAIN_LOG" | tail -3
    else
        echo "WARNING: No val_loss found in logs"
    fi
else
    echo "Test Training FAILED with exit code: $EXIT_CODE"
    echo "Check logs: $TRAIN_LOG"
fi
echo "=========================================="

# Cleanup symlinks
echo ""
echo "Test data location: $TEST_TILEDB_DIR (symlinks)"
echo "Output location: $OUTPUT_DIR"
