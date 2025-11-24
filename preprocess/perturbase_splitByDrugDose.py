#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import scanpy as sc
import pandas as pd
from collections import defaultdict
import warnings

def extract_drug_name_with_dose(gene_value):
    """从gene值中提取药物名称，保留剂量信息但进行文件名安全处理"""
    # 如果gene值是CTRL，直接返回None
    if gene_value == 'CTRL':
        return 'CTRL'
    if gene_value == 'None':
        return 'None'

    # 分割逗号分隔的值
    drugs = str(gene_value).split(',')
    
    # 对每个药物名称进行处理
    cleaned_drugs = []
    for drug in drugs:
        drug_cleaned = drug.strip()
        
        # 保留剂量信息，但替换特殊字符使其适合文件名
        # 将μg/ml替换为ug_ml
        drug_cleaned = drug_cleaned.replace('μg/ml', 'ug_ml')
        # 将μM替换为uM
        drug_cleaned = drug_cleaned.replace('μM', 'uM')
        # 将其他特殊字符（除了点、连字符和字母数字）替换为下划线
        drug_cleaned = re.sub(r'[^\w\.-]', '_', drug_cleaned)
        
        cleaned_drugs.append(drug_cleaned)
    
    # 将药物列表转换为下划线连接的字符串
    return '_'.join(sorted(cleaned_drugs))

def split_h5ad_by_gene_with_dose(input_file, output_prefix, min_cells=500):
    """根据gene列拆分h5ad文件，保留剂量信息"""
    
    # 读取h5ad文件，忽略观察名不唯一的警告
    print(f"正在读取文件: {input_file}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adata = sc.read_h5ad(input_file)
    
    # 确保观察名是唯一的
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    
    # 检查gene列是否存在
    if 'gene' not in adata.obs.columns:
        raise ValueError("数据集中没有找到'gene'列")
    
    # 收集所有可能的药物组合
    drug_combinations = defaultdict(list)
    
    # 首先获取所有CTRL细胞的索引
    ctrl_indices = []
    for idx, gene_value in adata.obs['gene'].items():
        if gene_value == 'CTRL':
            ctrl_indices.append(idx)
    
    print(f"找到 {len(ctrl_indices)} 个CTRL细胞")
    
    # 分析每个样本的gene值，找出所有非CTRL的药物组合
    for idx, gene_value in adata.obs['gene'].items():
        if gene_value != 'CTRL':
            drug_str = extract_drug_name_with_dose(gene_value)
            if drug_str:
                drug_combinations[drug_str].append(idx)
    
    print(f"找到 {len(drug_combinations)} 个药物组合")
    
    # 为每个组合创建子集并保存
    saved_files = []
    skipped_due_to_min_cells = 0
    
    for drug_str, indices in drug_combinations.items():
        # 检查药物细胞数量是否满足最小细胞数要求
        if len(indices) < min_cells:
            print(f"跳过 {drug_str}: 只有 {len(indices)} 个药物细胞 (< {min_cells})")
            skipped_due_to_min_cells += 1
            continue
        
        # 创建子集：包含该药物组合的所有细胞和所有CTRL细胞
        # 使用集合去重，然后转换为列表
        all_indices_set = set(indices) | set(ctrl_indices)
        all_indices = list(all_indices_set)
        
        # 创建子集
        subset = adata[all_indices].copy()
        
        # 确保子集的观察名是唯一的
        subset.obs_names_make_unique()
        adata.var_names_make_unique()

        # 添加time列：CTRL为0，非CTRL为1
        subset.obs['time'] = 0  # 默认为0
        for idx in indices:
            if idx in subset.obs.index:
                subset.obs.at[idx, 'time'] = 1
        
        # 生成输出文件名 - 使用前缀.药物名称.time.h5ad格式
        output_filename = f"{output_prefix}.{drug_str}.time.h5ad"
        
        # 保存文件，忽略观察名不唯一的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            subset.write(output_filename)
        
        saved_files.append(output_filename)
        
        print(f"保存: {output_filename} (包含 {len(all_indices)} 个细胞，其中CTRL: {len(ctrl_indices)}, {drug_str}: {len(indices)})")
    
    # 生成汇总信息
    print(f"\n拆分完成!")
    print(f"共生成 {len(saved_files)} 个文件")
    print(f"跳过 {skipped_due_to_min_cells} 个药物组合 (细胞数 < {min_cells})")
    print("生成的文件:")
    for filename in saved_files:
        print(f"  - {filename}")

def main():
    parser = argparse.ArgumentParser(description='根据h5ad文件中的gene列拆分成多个文件，保留剂量信息')
    parser.add_argument('input_file', help='输入的h5ad文件路径')
    parser.add_argument('output_prefix', help='输出文件前缀，格式为"前缀.药物名称.time.h5ad"')
    parser.add_argument('--min_cells', type=int, default=500, help='最小药物细胞数阈值，默认500')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    try:
        split_h5ad_by_gene_with_dose(args.input_file, args.output_prefix, args.min_cells)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()
