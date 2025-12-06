import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import tiledb
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import json
import gc
import scipy.sparse as sp
from typing import List, Dict, Optional, Tuple
import threading
import queue
import time

# 忽略 Scanpy 的一些 FutureWarning
warnings.filterwarnings("ignore")

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gene_vocab(vocab_path: str) -> List[str]:
    """加载基因词表"""
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with open(path, 'r') as f:
        genes = [line.strip() for line in f if line.strip()]
    return genes

def process_h5ad_vectorized(args) -> Optional[Dict]:
    """
    Worker 函数：处理单个 h5ad 文件
    """
    # 【优化】接收预构建好的 gene_map，而不是 list
    file_path, target_gene_map, target_genes_list, min_genes, target_sum, is_ood_flag = args
    
    try:
        # 1. 读取数据
        adata = sc.read_h5ad(file_path)

        # 2. 统一基因名为索引
        if "gene_symbols" in adata.var.columns:
            adata.var_names = adata.var["gene_symbols"].astype(str)
        adata.var_names_make_unique()

        # 3. 过滤细胞
        if min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)

        if adata.shape[0] == 0:
            return None

        # 4. 归一化 (注意：这里会改变数据为 log1p)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

        # --- 核心向量化逻辑 ---
        
        # A. 找到交集基因 (利用 set 转换 list 加速 isin)
        mask = np.isin(adata.var_names, target_genes_list)
        
        # B. 切片
        adata_sub = adata[:, mask]
        
        if adata_sub.shape[1] == 0:
            return None 
            
        # C. 转 COO
        X_coo = adata_sub.X.tocoo()
        
        # D. 索引重映射
        sub_gene_names = adata_sub.var_names
        
        # 使用传入的 map 进行映射
        local_to_global = np.array([target_gene_map[g] for g in sub_gene_names], dtype=np.int64)
        new_gene_indices = local_to_global[X_coo.col]
        
        # 构建结果
        res = {
            'n_cells': adata.shape[0],
            'row_indices': X_coo.row.astype(np.int64),
            'col_indices': new_gene_indices,
            'values': X_coo.data.astype(np.float32),
            'is_ood': is_ood_flag,
            'file_path': str(file_path)
        }
        
        # 【安全】主动释放大对象内存
        del adata, adata_sub, X_coo
        gc.collect()
        
        return res

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def init_tiledb_array(output_dir: Path, n_genes: int):
    tiledb_path = output_dir / "all_data"
    
    if tiledb_path.exists():
        import shutil
        logger.warning(f"Output path exists, cleaning up: {tiledb_path}")
        shutil.rmtree(tiledb_path)
    tiledb_path.mkdir(parents=True)
    
    # 瓦片大小 10万
    tile_extent = 100000
    # 防止 int64 溢出
    max_domain = np.iinfo(np.int64).max - tile_extent - 1000
    
    # 1. Counts Schema
    counts_uri = str(tiledb_path / "counts")
    dom = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, max_domain), tile=tile_extent, dtype=np.int64),
        tiledb.Dim(name="gene_index", domain=(0, n_genes - 1), tile=n_genes, dtype=np.int64),
    )
    schema = tiledb.ArraySchema(
        domain=dom, sparse=True, attrs=[tiledb.Attr(name="data", dtype=np.float32)], allows_duplicates=False,
    )
    tiledb.Array.create(counts_uri, schema)
    
    # 2. Metadata Schema
    meta_uri = str(tiledb_path / "cell_metadata")
    dom_meta = tiledb.Domain(
        tiledb.Dim(name="cell_index", domain=(0, max_domain), tile=tile_extent, dtype=np.int64)
    )
    schema_meta = tiledb.ArraySchema(
        domain=dom_meta, sparse=True,
        attrs=[
            tiledb.Attr(name="is_ood", dtype=np.int8),
            tiledb.Attr(name="file_source", dtype='ascii', var=True) 
        ]
    )
    tiledb.Array.create(meta_uri, schema_meta)
    
    return tiledb_path

class AsyncBatchWriter:
    def __init__(self, tiledb_path: Path, batch_size: int = 500000):
        self.counts_uri = str(tiledb_path / "counts")
        self.meta_uri = str(tiledb_path / "cell_metadata")
        self.batch_size = batch_size
        self.global_cell_offset = 0
        
        self.current_buffer = self._init_buffer()
        self.current_count = 0
        
        # 【重要修复】设置 maxsize 防止内存爆炸 (背压机制)
        # 允许队列里存 3 个 batch，如果满了，主进程的 add() 会阻塞等待
        self.write_queue = queue.Queue(maxsize=3)
        
        self.is_running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        self.write_error = None

    def _init_buffer(self):
        return {
            'rows': [], 'cols': [], 'vals': [],
            'meta_indices': [], 'meta_ood': [], 'meta_src': []
        }

    def add(self, result: Dict):
        if self.write_error: raise self.write_error
        if not result: return
            
        n_cells = result['n_cells']
        global_rows = result['row_indices'] + self.global_cell_offset
        
        self.current_buffer['rows'].append(global_rows)
        self.current_buffer['cols'].append(result['col_indices'])
        self.current_buffer['vals'].append(result['values'])
        self.current_buffer['meta_indices'].append(np.arange(self.global_cell_offset, self.global_cell_offset + n_cells))
        self.current_buffer['meta_ood'].append(np.full(n_cells, result['is_ood'], dtype=np.int8))
        self.current_buffer['meta_src'].extend([result['file_path']] * n_cells)
        
        self.global_cell_offset += n_cells
        self.current_count += n_cells
        
        if self.current_count >= self.batch_size:
            self._push_to_queue()

    def _push_to_queue(self):
        if self.current_count == 0: return
        logger.info(f"  [Main] Batch full ({self.current_count} cells). Pushing to queue (Size: {self.write_queue.qsize()})...")
        
        task = (self.current_buffer, self.current_count)
        # put 默认是阻塞的，如果队列满了，这里会停下等待，保护内存
        self.write_queue.put(task)
        
        self.current_buffer = self._init_buffer()
        self.current_count = 0

    def _writer_loop(self):
        while self.is_running or not self.write_queue.empty():
            try:
                try:
                    buffer, count = self.write_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                t0 = time.time()
                # 稍微提高 TileDB 内部并发，但不要太高，因为已经在后台线程了
                cfg = tiledb.Config({"sm.compute_concurrency_level": "4", "sm.io_concurrency_level": "4"})
                ctx = tiledb.Ctx(cfg)
                
                # 写入 Counts
                all_rows = np.concatenate(buffer['rows'])
                all_cols = np.concatenate(buffer['cols'])
                all_vals = np.concatenate(buffer['vals'])
                with tiledb.open(self.counts_uri, 'w', ctx=ctx) as arr:
                    arr[all_rows, all_cols] = all_vals
                
                # 写入 Metadata
                with tiledb.open(self.meta_uri, 'w', ctx=ctx) as arr:
                    arr[np.concatenate(buffer['meta_indices'])] = {
                        'is_ood': np.concatenate(buffer['meta_ood']),
                        'file_source': np.array(buffer['meta_src'], dtype=object)
                    }
                
                del buffer, all_rows, all_cols, all_vals
                gc.collect()
                
                logger.info(f"  [Async] Write finished. Time: {time.time()-t0:.1f}s.")
                self.write_queue.task_done()
                
            except Exception as e:
                logger.error(f"Async Writer Crashed: {e}")
                self.write_error = e
                break

    def finish(self):
        self._push_to_queue()
        self.is_running = False
        self.writer_thread.join()
        if self.write_error: raise self.write_error

def main():
    parser = argparse.ArgumentParser(description="Efficient TileDB Converter (Single Array + OOD Flag)")
    parser.add_argument("--csv_path", type=str, default="data/data_info/ae_data_info.csv")
    parser.add_argument("--vocab_path", type=str, default="data/data_info/gene_order.tsv")
    parser.add_argument("--output_dir", type=str, default="data/processed_data/tiledb_unified")
    parser.add_argument("--min_genes", type=int, default=200)
    parser.add_argument("--target_sum", type=float, default=1e4)
    parser.add_argument("--num_workers", type=int, default=256) 
    parser.add_argument("--batch_size", type=int, default=1000000)
    
    # 【新增】限制处理的文件数量，方便测试
    # 默认为 0 或 None 表示处理所有文件
    parser.add_argument("--max_files", type=int, default=5000, 
                        help="Limit number of files to process. Set to 0 for all files.")
    
    args = parser.parse_args()
    
    # 1. 准备
    target_genes = load_gene_vocab(args.vocab_path)
    target_gene_map = {g: i for i, g in enumerate(target_genes)}
    
    info_df = pd.read_csv(args.csv_path)
    
    # 【新增】文件数量截断逻辑
    total_files_found = len(info_df)
    if args.max_files > 0:
        info_df = info_df.head(args.max_files)
        logger.warning(f"!!! DEBUG MODE !!! Limiting processing to first {len(info_df)} files (Found {total_files_found} total).")
    else:
        logger.info(f"Processing ALL {total_files_found} files found in CSV.")

    output_dir = Path(args.output_dir)
    logger.info(f"Targets: {len(target_genes)} genes.")
    
    # 2. 初始化
    tiledb_path = init_tiledb_array(output_dir, len(target_genes))
    
    # 写入基因注释
    gene_annot_uri = str(tiledb_path / "gene_annotation")
    tiledb.Array.create(gene_annot_uri, tiledb.ArraySchema(
        domain=tiledb.Domain(tiledb.Dim(name="gene_index", domain=(0, len(target_genes)-1), tile=len(target_genes), dtype=np.int64)),
        sparse=False,
        attrs=[tiledb.Attr(name="gene_symbol", dtype='ascii', var=True)]
    ))
    with tiledb.open(gene_annot_uri, 'w') as arr:
        arr[:] = {'gene_symbol': np.array(target_genes, dtype=object)}

    # 3. 准备任务
    tasks = []
    for _, row in info_df.iterrows():
        is_ood = int(row.get('full_validation_dataset', 0))
        tasks.append((
            row['file_path'], 
            target_gene_map,
            target_genes,
            args.min_genes, 
            args.target_sum, 
            is_ood
        ))

    # 4. 执行
    writer = AsyncBatchWriter(tiledb_path, batch_size=args.batch_size)
    logger.info(f"Starting parallel processing with {args.num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_h5ad_vectorized, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing & Writing"):
            file_name = futures[future]
            try:
                result = future.result()
                if result:
                    writer.add(result)
            except Exception as e:
                logger.error(f"Failed to process {file_name}: {e}")

    writer.finish()
    
    # 5. 保存 Metadata
    metadata = {
        'total_cells': writer.global_cell_offset,
        'n_genes': len(target_genes),
        'storage_path': str(tiledb_path),
        'files_processed': len(info_df) # 记录一下实际处理了多少文件
    }
    with open(tiledb_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("="*50)
    logger.info(f"Done! Total cells stored: {writer.global_cell_offset}")
    logger.info("Don't forget to run consolidation script afterwards!")

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()