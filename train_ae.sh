#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,3,2,4,5,6,7

# 1. 强制禁用 Infiniband (因为它们全是 DOWN 的，这是报错的主因)
export NCCL_IB_DISABLE=1

# 2. 指定使用第一个正常工作的以太网口 (enp83s0f0)
export NCCL_SOCKET_IFNAME=enp83s0f0


# 1. 禁用 NVLS (解决当前的 transport/nvls.cc 报错)
export NCCL_NVLS_ENABLE=0

# 2. 禁用 Infiniband (解决之前的 IB Down 问题)
export NCCL_IB_DISABLE=1

# 3. 指定网卡 (解决之前的 IP 问题)
export NCCL_SOCKET_IFNAME=enp83s0f0

export NCCL_TIMEOUT=300
# Autoencoder Training Script (Refactored)
#
# Workflow:
# 1. Preprocess H5AD files -> Parquet Shards (Train/Val/Test/OOD)
# 2. Train Autoencoder on Parquet data
#
# Usage: ./train_ae.sh [DATA_DIR] [VOCAB_FILE] [OUTPUT_DIR]

DATA_DIR=${1:-"/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens"}
VOCAB_FILE=${2:-"data_info/gene_order.tsv"}
OUTPUT_DIR=${3:-"output/ae_large_scale/version3_larger"}
CSV_INFO="data_info/ae_data_info.csv"

PROCESSED_DIR="${DATA_DIR}/.parquet"

echo "=================================================="
echo "Starting Autoencoder Workflow"
echo "Data Directory: $DATA_DIR"
echo "Vocabulary File: $VOCAB_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Processed Data: $PROCESSED_DIR"
echo "=================================================="

# Ensure directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROCESSED_DIR"

# Step 1: Preprocessing
echo ""
echo "[Step 1] Preprocessing Data..."
echo "=================================================="

# Check if data is already processed (check for train parquet file)
if [ -f "$PROCESSED_DIR/train.parquet" ]; then
    echo "⚠️  Processed parquet files found in $PROCESSED_DIR"
    echo ""
    ls -lh "$PROCESSED_DIR"/*.parquet 2>/dev/null || true
    echo ""
    read -p "Use existing parquet files? [Y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Removing existing files and reprocessing..."
        rm -f "$PROCESSED_DIR"/*.parquet
        rm -f "$PROCESSED_DIR"/metadata.json
    else
        echo "Skipping preprocessing, using existing files."
        echo ""
        SKIP_PREPROCESSING=true
    fi
else
    echo "No processed parquet files found. Starting preprocessing..."
fi

if [ "$SKIP_PREPROCESSING" != "true" ]; then
    echo ""
    echo "Starting preprocessing with OPTIMIZED architecture..."
    echo "Key improvements:"
    echo "  - Workers write temp files directly (no DataFrame serialization)"
    echo "  - Main process only merges file paths (eliminates 100GB+ IPC overhead)"
    echo "  - Expected speedup: 3-5x faster"
    echo ""
    echo "Parameters:"
    echo "  - Min genes: 200"
    echo "  - Workers: 150 (充分利用224核，避免GPFS过载)"
    echo "  - Batch size: 500 files per batch"
    echo "  - Files per worker batch: 25 (批量处理减少任务调度)"
    echo "  - Output: Sharded parquet files per split"
    echo ""

    python preprocess_ae.py \
        --csv_path "$CSV_INFO" \
        --vocab_path "$VOCAB_FILE" \
        --output_dir "$PROCESSED_DIR" \
        --min_genes 200 \
        --num_workers 202 \
        --batch_size 1000 \
        --files_per_worker_batch 40

    PREPROCESS_EXIT_CODE=$?

    if [ $PREPROCESS_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "✗ Preprocessing failed with exit code: $PREPROCESS_EXIT_CODE"
        echo "  Check the output above for errors."
        exit $PREPROCESS_EXIT_CODE
    fi

    echo ""
    echo "✓ Preprocessing completed successfully!"
    echo ""
fi

# Step 1.5: Check and generate metadata cache for multi-worker loading
echo ""
echo "[Step 1.5] Checking metadata cache for DataLoader..."
echo "=================================================="

CACHE_MISSING=false
for split in train val test ood; do
    CACHE_FILE="$PROCESSED_DIR/${split}_shards/metadata_cache.json"
    if [ ! -f "$CACHE_FILE" ]; then
        echo "⚠️  Missing cache: ${split}_shards/metadata_cache.json"
        CACHE_MISSING=true
    fi
done

if [ "$CACHE_MISSING" = true ]; then
    echo ""
    echo "Generating metadata caches for fast multi-worker loading..."
    python3 << EOF
import json
import pyarrow.parquet as pq
from pathlib import Path

processed_dir = Path("$PROCESSED_DIR")

for split_name in ['train', 'val', 'test', 'ood']:
    shard_dir = processed_dir / f"{split_name}_shards"
    if not shard_dir.exists():
        continue

    cache_file = shard_dir / "metadata_cache.json"
    if cache_file.exists():
        continue

    files = sorted(list(shard_dir.glob("*.parquet")))
    if not files:
        continue

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

    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"  ✓ {split_name}: {cache['n_cells']:,} cells, {len(files)} shards")

print("✅ Metadata caches ready!")
EOF
else
    echo "✓ All metadata caches found"
fi

echo ""

# Step 2: Training
echo ""
echo "[Step 2] Training Autoencoder..."
echo "=================================================="

# Create log directory
mkdir -p "$OUTPUT_DIR/logs"

# Define log files with timestamp
TRAIN_LOG="$OUTPUT_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "================================================================================"
echo "Training logs will be saved to:"
echo "  $TRAIN_LOG"
echo ""
echo "To monitor training in real-time, open a new terminal and run:"
echo "  tail -f $TRAIN_LOG"
echo ""
echo "For debug logs from each GPU rank:"
echo "  tail -f $OUTPUT_DIR/logs/debug_rank*.log"
echo "================================================================================"
echo ""
echo "Starting training (output redirected to log file)..."

# Note: Updated paths to point to sharded parquet directories
# Now using metadata cache for fast multi-worker loading
# Redirect all output to log file
python train_ae.py \
    --config_path="config/ae.yaml" \
    --data.dataset_type="parquet" \
    --data.processed_path.train="$PROCESSED_DIR/train_shards" \
    --data.processed_path.val="$PROCESSED_DIR/val_shards" \
    --data.processed_path.test="$PROCESSED_DIR/test_shards" \
    --data.processed_path.ood="$PROCESSED_DIR/ood_shards" \
    --logging.output_dir="$OUTPUT_DIR" \
    --training.max_epochs=1 \
    --accelerator.precision="16-mixed" \
    --data.num_workers=2 \
    --debug=false \
    >> "$TRAIN_LOG" 2>&1

TRAIN_EXIT_CODE=$?

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "  Check logs for details: $TRAIN_LOG"
fi
echo ""
echo "Full logs saved to: $TRAIN_LOG"
echo "Workflow finished."
