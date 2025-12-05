#!/bin/bash
# TileDB Autoencoder Training Script
#
# Usage:
#   ./train_ae_tiledb.sh                           # Use default paths
#   ./train_ae_tiledb.sh /path/to/tiledb output/   # Custom paths

set -e

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL Settings for multi-GPU
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp83s0f0
export NCCL_NVLS_ENABLE=0
export NCCL_TIMEOUT=1800

# Paths
TILEDB_DIR="${1:-data/ae_tiledb}"
OUTPUT_DIR="${2:-output/ae_tiledb}"
VOCAB_FILE="data_info/gene_order.tsv"
CSV_INFO="data_info/ae_data_info.csv"

echo "=========================================="
echo "TileDB Autoencoder Training"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  TileDB Data:    $TILEDB_DIR"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  GPUs:           $CUDA_VISIBLE_DEVICES"
echo ""

# Check if TileDB data exists
if [ ! -d "$TILEDB_DIR/train_tiledb" ]; then
    echo "TileDB data not found at: $TILEDB_DIR/train_tiledb"
    echo ""
    echo "Run preprocessing first:"
    echo "  ./preprocess_tiledb.sh $TILEDB_DIR"
    echo ""

    read -p "Run preprocessing now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting preprocessing..."
        ./preprocess_tiledb.sh "$TILEDB_DIR"
    else
        exit 1
    fi
fi

# Show TileDB data info
echo "TileDB datasets:"
for split in train val test ood; do
    tiledb_dir="$TILEDB_DIR/${split}_tiledb"
    if [ -d "$tiledb_dir" ] && [ -f "$tiledb_dir/metadata.json" ]; then
        n_cells=$(python3 -c "import json; print(f\"{json.load(open('$tiledb_dir/metadata.json'))['n_cells']:,}\")" 2>/dev/null || echo "?")
        sparsity=$(python3 -c "import json; print(f\"{json.load(open('$tiledb_dir/metadata.json'))['sparsity']:.1%}\")" 2>/dev/null || echo "?")
        echo "  ${split}: $n_cells cells ($sparsity sparse)"
    fi
done
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR/logs"

# Log file
TRAIN_LOG="$OUTPUT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Starting Training..."
echo "=========================================="
echo ""
echo "Logs: $TRAIN_LOG"
echo "Monitor: tail -f $TRAIN_LOG"
echo ""

# Train with TileDB
python train_ae.py \
    --config_path="config/ae.yaml" \
    --data.dataset_type="tiledb" \
    --data.processed_path.train="$TILEDB_DIR/train_tiledb" \
    --data.processed_path.val="$TILEDB_DIR/val_tiledb" \
    --data.processed_path.test="$TILEDB_DIR/test_tiledb" \
    --data.processed_path.ood="$TILEDB_DIR/ood_tiledb" \
    --logging.output_dir="$OUTPUT_DIR" \
    --training.max_epochs=50 \
    --training.batch_size=4096 \
    --accelerator.precision="16-mixed" \
    --data.num_workers=0 \
    --debug=false \
    2>&1 | tee "$TRAIN_LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training COMPLETE!"
    echo ""
    echo "Checkpoints: $OUTPUT_DIR/checkpoints/"
    ls -lh "$OUTPUT_DIR/checkpoints/" 2>/dev/null | head -5
else
    echo "Training FAILED with exit code: $EXIT_CODE"
    echo "Check logs: $TRAIN_LOG"
fi
echo "=========================================="
