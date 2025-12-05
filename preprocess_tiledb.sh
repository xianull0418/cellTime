#!/bin/bash
# Preprocess h5ad files directly to TileDB format
# No intermediate Parquet step - similar to CellArr approach
#
# Usage:
#   ./preprocess_tiledb.sh                    # Use default paths
#   ./preprocess_tiledb.sh /path/to/output    # Custom output directory

set -e

# Configuration
CSV_PATH="data_info/ae_data_info.csv"
VOCAB_PATH="data_info/gene_order.tsv"
OUTPUT_DIR="${1:-data/ae_tiledb}"
NUM_WORKERS=64
MIN_GENES=200

echo "=========================================="
echo "Preprocess h5ad → TileDB (Direct)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  CSV Path:          $CSV_PATH"
echo "  Vocab Path:        $VOCAB_PATH"
echo "  Output Directory:  $OUTPUT_DIR"
echo "  Num Workers:       $NUM_WORKERS"
echo "  Min Genes:         $MIN_GENES"
echo ""

# Check if input files exist
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found: $CSV_PATH"
    echo ""
    echo "Please create a CSV file with columns:"
    echo "  - file_path: path to h5ad file"
    echo "  - full_validation_dataset: 1 for OOD, 0 for train/val/test split"
    exit 1
fi

if [ ! -f "$VOCAB_PATH" ]; then
    echo "ERROR: Vocabulary file not found: $VOCAB_PATH"
    echo ""
    echo "Please create a gene vocabulary file (one gene per line)"
    exit 1
fi

# Count input files
TOTAL_FILES=$(wc -l < "$CSV_PATH")
TOTAL_FILES=$((TOTAL_FILES - 1))  # Subtract header
echo "Found $TOTAL_FILES h5ad files to process"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create log file
LOG_FILE="$OUTPUT_DIR/preprocess_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

echo "=========================================="
echo "Starting preprocessing..."
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Read h5ad files and normalize (log1p)"
echo "  2. Align genes to vocabulary"
echo "  3. Split into train/val/test/ood (90/5/5%)"
echo "  4. Write directly to TileDB sparse arrays"
echo ""
echo "No intermediate Parquet files - direct TileDB build!"
echo ""

# Run preprocessing
python preprocess_tiledb.py \
    --csv_path="$CSV_PATH" \
    --vocab_path="$VOCAB_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --num_workers=$NUM_WORKERS \
    --min_genes=$MIN_GENES \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Preprocessing COMPLETE!"
    echo ""
    echo "Output structure:"
    echo ""

    # Show directory structure
    if [ -d "$OUTPUT_DIR" ]; then
        for split in train val test ood; do
            tiledb_dir="$OUTPUT_DIR/${split}_tiledb"
            if [ -d "$tiledb_dir" ]; then
                size=$(du -sh "$tiledb_dir" 2>/dev/null | cut -f1)
                if [ -f "$tiledb_dir/metadata.json" ]; then
                    n_cells=$(python3 -c "import json; print(json.load(open('$tiledb_dir/metadata.json'))['n_cells'])" 2>/dev/null || echo "?")
                    sparsity=$(python3 -c "import json; print(f\"{json.load(open('$tiledb_dir/metadata.json'))['sparsity']:.1%}\")" 2>/dev/null || echo "?")
                    echo "  ${split}_tiledb/: $n_cells cells, $sparsity sparse, $size"
                else
                    echo "  ${split}_tiledb/: $size"
                fi
            fi
        done
    fi

    echo ""
    echo "To train with TileDB data:"
    echo ""
    echo "  python train_ae.py \\"
    echo "      --data.dataset_type=tiledb \\"
    echo "      --data.processed_path.train=$OUTPUT_DIR/train_tiledb \\"
    echo "      --data.processed_path.val=$OUTPUT_DIR/val_tiledb \\"
    echo "      --data.processed_path.ood=$OUTPUT_DIR/ood_tiledb"
    echo ""
else
    echo "Preprocessing FAILED with exit code: $EXIT_CODE"
    echo "Check log file: $LOG_FILE"
fi
echo "=========================================="
