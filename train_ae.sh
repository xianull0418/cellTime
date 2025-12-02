#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

# echo "=================================================="
# echo "Starting Autoencoder Workflow"
# echo "Data Directory: $DATA_DIR"
# echo "Vocabulary File: $VOCAB_FILE"
# echo "Output Directory: $OUTPUT_DIR"
# echo "Processed Data: $PROCESSED_DIR"
# echo "=================================================="

# Ensure directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROCESSED_DIR"

# Step 1: Preprocessing
echo "[Step 1] Preprocessing Data..."
# Check if data is already processed (check for train shards directory)
if [ ! -d "$PROCESSED_DIR/train_shards" ]; then
    # Added --num_workers optimization
    python preprocess_ae.py \
        --csv_path "$CSV_INFO" \
        --vocab_path "$VOCAB_FILE" \
        --output_dir "$PROCESSED_DIR" \
        --min_genes 200 \
        --num_workers 64 \
        --format "parquet"
else
    echo "Processed shards found in $PROCESSED_DIR/train_shards. Skipping preprocessing."
    echo "Remove directories to force re-processing."
fi

# Step 2: Training
echo "[Step 2] Training Autoencoder..."

# Note: Updated paths to point to shard DIRECTORIES
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
    --data.num_workers=4

echo "Workflow finished."
