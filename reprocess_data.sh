#!/bin/bash
# 重新预处理数据脚本
# 删除旧的不均匀数据，生成新的均匀大小的 shard 文件

DATA_DIR="/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens"
PARQUET_DIR="${DATA_DIR}/.parquet"

echo "=========================================="
echo "重新预处理单细胞数据"
echo "=========================================="
echo ""

# 显示旧数据大小
echo "📊 当前数据大小："
du -sh ${PARQUET_DIR}/*_shards 2>/dev/null || echo "没有找到旧数据"
echo ""

# 询问确认
read -p "⚠️  确认删除以上目录并重新处理数据？[y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🗑️  删除旧数据..."
rm -rf ${PARQUET_DIR}/train_shards
rm -rf ${PARQUET_DIR}/val_shards
rm -rf ${PARQUET_DIR}/test_shards
rm -rf ${PARQUET_DIR}/ood_shards
rm -rf ${PARQUET_DIR}/temp_chunks

echo "✓ 旧数据已删除"
echo ""

# 重新运行预处理
echo "🔄 开始重新预处理..."
echo "参数说明："
echo "  - shard_size: 8000 (每个文件 8000 个样本)"
echo "  - num_workers: 64 (并行处理)"
echo "  - format: parquet"
echo ""

python preprocess_ae.py \
    --csv_path "data_info/ae_data_info.csv" \
    --vocab_path "data_info/gene_order.tsv" \
    --output_dir "${PARQUET_DIR}" \
    --min_genes 200 \
    --num_workers 64 \
    --shard_size 8000 \
    --format "parquet"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 预处理完成！"
    echo ""
    echo "📊 新数据统计："
    for split in train val test ood; do
        shard_dir="${PARQUET_DIR}/${split}_shards"
        if [ -d "$shard_dir" ]; then
            num_files=$(ls -1 ${shard_dir}/*.parquet 2>/dev/null | wc -l)
            total_size=$(du -sh ${shard_dir} | cut -f1)
            echo "  ${split}: ${num_files} 个文件, 总大小 ${total_size}"
        fi
    done
    echo ""
    echo "验证文件大小均匀性..."
    python3 << 'EOF'
import pyarrow.parquet as pq
from pathlib import Path
import statistics

parquet_dir = Path("/gpfs/hybrid/data/downloads/gcloud/arc-scbasecount/2025-02-25/h5ad/GeneFull_Ex50pAS/Homo_sapiens/.parquet")
train_files = list((parquet_dir / "train_shards").glob("*.parquet"))

if train_files:
    # 采样检查
    sample_files = train_files[:50]
    sizes = []
    for f in sample_files:
        meta = pq.read_metadata(f)
        sizes.append(meta.num_rows)

    print(f"  采样 {len(sample_files)} 个文件:")
    print(f"    最小样本数: {min(sizes):,}")
    print(f"    最大样本数: {max(sizes):,}")
    print(f"    平均样本数: {statistics.mean(sizes):,.0f}")
    print(f"    标准差: {statistics.stdev(sizes):,.0f}")
    print(f"    差异倍数: {max(sizes)/min(sizes):.2f}x")

    if max(sizes) / min(sizes) < 1.1:
        print("  ✅ 文件大小非常均匀！")
    else:
        print("  ⚠️  仍有一些大小差异")
EOF
else
    echo "❌ 预处理失败，退出码: $EXIT_CODE"
    echo "请查看日志排查问题"
    exit $EXIT_CODE
fi
