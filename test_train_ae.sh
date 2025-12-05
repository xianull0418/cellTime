#!/bin/bash
# Test Training Script for Autoencoder
# Uses subset of data for quick testing:
#   - Train: first 100 files
#   - Val: all 5 files
#   - Test: first 3 files
#   - OOD: first 10 files

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL Settings
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp83s0f0
export NCCL_NVLS_ENABLE=0
export NCCL_TIMEOUT=1800  # 30 minutes, increase for large data loading variance

# Paths
DATA_DIR="/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/.parquet"
OUTPUT_DIR=${1:-"output/ae_test_10epochs"}
TEST_DATA_DIR="data/ae_test_subset"

echo "=========================================="
echo "Test Autoencoder Training"
echo "=========================================="
echo ""
echo "Creating subset of data for testing..."
echo "  Train: 100 files (out of 105)"
echo "  Val:   5 files (all)"
echo "  Test:  3 files (out of 6)"
echo "  OOD:   10 files (out of 50)"
echo ""

# Clean and create test data directory
rm -rf "$TEST_DATA_DIR"
mkdir -p "$TEST_DATA_DIR"/{train_shards,val_shards,test_shards,ood_shards}

# Create symbolic links for train (first 100 files)
echo "Linking train files..."
train_files=("$DATA_DIR"/train_shards/*.parquet)
for i in {0..99}; do
    if [ -f "${train_files[$i]}" ]; then
        ln -s "${train_files[$i]}" "$TEST_DATA_DIR/train_shards/"
    fi
done

# Create symbolic links for val (all 5 files)
echo "Linking val files..."
for f in "$DATA_DIR"/val_shards/*.parquet; do
    if [ -f "$f" ]; then
        ln -s "$f" "$TEST_DATA_DIR/val_shards/"
    fi
done

# Create symbolic links for test (first 3 files)
echo "Linking test files..."
test_files=("$DATA_DIR"/test_shards/*.parquet)
for i in {0..2}; do
    if [ -f "${test_files[$i]}" ]; then
        ln -s "${test_files[$i]}" "$TEST_DATA_DIR/test_shards/"
    fi
done

# Create symbolic links for ood (first 10 files)
echo "Linking ood files..."
ood_files=("$DATA_DIR"/ood_shards/*.parquet)
for i in {0..9}; do
    if [ -f "${ood_files[$i]}" ]; then
        ln -s "${ood_files[$i]}" "$TEST_DATA_DIR/ood_shards/"
    fi
done

echo ""
echo "Test data subset created:"
ls -lh "$TEST_DATA_DIR"/*/  | grep -E "^total|parquet"

echo ""
echo "Generating metadata caches for test subset..."
python3 << 'EOF'
import json
import pyarrow.parquet as pq
from pathlib import Path

test_data_dir = Path("data/ae_test_subset")
for split_name in ['train', 'val', 'test', 'ood']:
    shard_dir = test_data_dir / f"{split_name}_shards"
    if not shard_dir.exists():
        continue

    files = sorted(list(shard_dir.glob("*.parquet")))
    if not files:
        continue

    # Generate cache
    cache = {
        'n_files': len(files),
        'files_hash': len(files),
        'n_cells': 0,
        'shard_lengths': [],
        'shard_offsets': []
    }

    cumulative = 0
    for f in files:
        meta = pq.read_metadata(f)
        n_rows = meta.num_rows
        cache['shard_lengths'].append(n_rows)
        cache['shard_offsets'].append(cumulative)
        cumulative += n_rows

    cache['n_cells'] = cumulative

    # Write cache
    cache_file = shard_dir / "metadata_cache.json"
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"  ✓ {split_name}: {cache['n_cells']:,} cells, {len(files)} files")

print("\n✅ Metadata caches generated!")
EOF

echo ""
echo "=========================================="
echo "Starting training..."
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR/logs"

# Define log file
TRAIN_LOG="$OUTPUT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Logs will be saved to: $TRAIN_LOG"
echo ""
echo "To monitor in real-time:"
echo "  tail -f $TRAIN_LOG"
echo ""

# Run training with test data
python train_ae.py \
    --config_path="config/ae.yaml" \
    --data.dataset_type="parquet" \
    --data.processed_path.train="$TEST_DATA_DIR/train_shards" \
    --data.processed_path.val="$TEST_DATA_DIR/val_shards" \
    --data.processed_path.test="$TEST_DATA_DIR/test_shards" \
    --data.processed_path.ood="$TEST_DATA_DIR/ood_shards" \
    --logging.output_dir="$OUTPUT_DIR" \
    --training.max_epochs=10 \
    --accelerator.precision="16-mixed" \
    --data.num_workers=4 \
    --debug=false \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Test training completed successfully!"

    echo ""
    echo "Checking DDP synchronization..."
    grep "TRAINING END" "$OUTPUT_DIR/logs/debug_rank0_"*.log 2>/dev/null | tail -8 || echo "No debug logs found"
else
    echo "✗ Test training failed with exit code: $TRAIN_EXIT_CODE"
    echo "  Check logs: $TRAIN_LOG"
fi

echo ""
echo "Test data location: $TEST_DATA_DIR"
echo "Output location: $OUTPUT_DIR"
echo "=========================================="
