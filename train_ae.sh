#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# Autoencoder Training Script for Large-Scale Single-Cell Data
# 
# Usage: ./train_ae.sh [DATA_DIR] [VOCAB_FILE] [OUTPUT_DIR]
#
# Example:
#   ./train_ae.sh /gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens \
#                 /gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/gene_order.tsv \
#                 output/ae_large_scale

DATA_DIR=${1:-"/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens"}
VOCAB_FILE=${2:-"gene_order.tsv"}
OUTPUT_DIR=${3:-"output/ae_large_scale"}

echo "Starting Autoencoder Training..."
echo "Data Directory: $DATA_DIR"
echo "Vocabulary File: $VOCAB_FILE"
echo "Output Directory: $OUTPUT_DIR"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Run training
python train_ae.py \
    --config_path="config/ae.yaml" \
    --data.data_path="$DATA_DIR" \
    --data.vocab_path="$VOCAB_FILE" \
    --logging.output_dir="$OUTPUT_DIR" \
    --model.hidden_dim="[2048, 1024, 512]" \
    --model.latent_dim=256 \
    --model.dropout_rate=0.05 \
    --training.batch_size=2048 \
    --training.learning_rate=1e-4 \
    --training.max_epochs=50 \
    --accelerator.precision="16-mixed" \
    --data.num_workers=16

echo "Training finished."

