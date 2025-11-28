#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=7

# Autoencoder Training Script (Refactored)
# 
# Workflow:
# 1. Preprocess H5AD files -> Parquet (Train/Val/Test/OOD splits)
# 2. Train Autoencoder on Parquet data
#
# Usage: ./train_ae.sh [DATA_DIR] [VOCAB_FILE] [OUTPUT_DIR]

DATA_DIR=${1:-"/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens"}
VOCAB_FILE=${2:-"gene_order.tsv"}
OUTPUT_DIR=${3:-"output/ae_large_scale/scbank_run"}
CSV_INFO="ae_data_info.csv"

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
echo "[Step 1] Preprocessing Data..."
# Check if data is already processed to avoid re-running if not needed (optional)
# For now, we run it to ensure consistency as per request.
if [ ! -f "$PROCESSED_DIR/train.parquet" ]; then
    # Added --num_workers optimization
    python preprocess_ae.py \
        --csv_path "$CSV_INFO" \
        --vocab_path "$VOCAB_FILE" \
        --output_dir "$PROCESSED_DIR" \
        --min_genes 200 \
        --num_workers 64
else
    echo "Processed files found in $PROCESSED_DIR. Skipping preprocessing."
    echo "Remove files to force re-processing."
fi

# Step 2: Training
echo "[Step 2] Training Autoencoder..."

python train_ae.py \
    --config_path="config/ae.yaml" \
    --data.dataset_type="parquet" \
    --data.parquet_path.train="$PROCESSED_DIR/train.parquet" \
    --data.parquet_path.val="$PROCESSED_DIR/val.parquet" \
    --data.parquet_path.test="$PROCESSED_DIR/test.parquet" \
    --data.parquet_path.ood="$PROCESSED_DIR/ood.parquet" \
    --logging.output_dir="$OUTPUT_DIR" \
    --model.hidden_dim="[2048, 1024, 512]" \
    --model.latent_dim=256 \
    --model.dropout_rate=0.05 \
    --training.batch_size=2048 \
    --training.learning_rate=1e-4 \
    --training.max_epochs=50 \
    --accelerator.precision="16-mixed" \
    --data.num_workers=16

echo "Workflow finished."
