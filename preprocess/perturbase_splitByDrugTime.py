#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import scanpy as sc
import pandas as pd
from collections import defaultdict
import warnings

def extract_drug_names(gene_value):
    """从gene值中提取药物名称，去除_Day数字"""
    # 分割逗号分隔的值
    drugs = str(gene_value).split(',')
    
    # 对每个药物名称进行处理
    cleaned_drugs = []
    for drug in drugs:
        # 多次替换确保去除所有_Day数字模式
        drug_cleaned = drug.strip()
        # 使用循环替换确保去除所有_Day模式
        while True:
            new_drug = re.sub(r'_Day\d*\.?\d*', '', drug_cleaned)
            if new_drug == drug_cleaned:
                break
            drug_cleaned = new_drug
        cleaned_drugs.append(drug_cleaned)
    
    # 返回所有非CTRL药物，用下划线连接
    non_ctrl_drugs = [d for d in cleaned_drugs if d != 'CTRL']
    return '_'.join(sorted(non_ctrl_drugs)) if non_ctrl_drugs else None

def process_time_column(adata):
    """处理time列，将Day去除并转换为浮点数"""
    if 'time' in adata.obs.columns:
        def convert_time(time_val):
            if pd.isna(time_val):
                return time_val
            time_str = str(time_val)
            # 去除Day前缀并转换为浮点数
            time_str = re.sub(r'^Day', '', time_str)
            try:
                return float(time_str)
            except ValueError:
                return time_val
        
        adata.obs['time'] = adata.obs['time'].apply(convert_time)
    return adata

def split_h5ad_by_gene(input_file, output_prefix):
    """根据gene列拆分h5ad文件"""
    
    # 读取h5ad文件
    print(f"正在读取文件: {input_file}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        adata = sc.read_h5ad(input_file)

    # 确保观察名是唯一的
    adata.obs_names_make_unique()
    
    # 检查gene列是否存在
    if 'gene' not in adata.obs.columns:
        raise ValueError("数据集中没有找到'gene'列")
    
    # 处理time列
    adata = process_time_column(adata)
    
    # 收集所有可能的药物组合
    drug_combinations = defaultdict(list)
    
    # 首先获取所有CTRL细胞的索引
    ctrl_indices = []
    for idx, gene_value in adata.obs['gene'].items():
        if 'CTRL' in str(gene_value):
            ctrl_indices.append(idx)
    
    # 分析每个样本的gene值，找出所有非CTRL的药物组合
    for idx, gene_value in adata.obs['gene'].items():
        drug_str = extract_drug_names(gene_value)
        
        # 如果包含非CTRL药物
        if drug_str:
            drug_combinations[drug_str].append(idx)
    
    print(f"找到 {len(drug_combinations)} 个药物组合")
    print(f"找到 {len(ctrl_indices)} 个CTRL细胞")
    
    # 为每个组合创建子集并保存
    saved_files = []
    for drug_str, indices in drug_combinations.items():
        if len(indices) > 0:
            # 创建子集：包含该药物组合的所有细胞和所有CTRL细胞
            # 使用集合去重，然后转换为列表
            all_indices_set = set(indices) | set(ctrl_indices)
            all_indices = list(all_indices_set)
            
            # 创建子集
            subset = adata[all_indices].copy()
            
            # 确保子集的观察名是唯一的
            subset.obs_names_make_unique()
            
            # 生成输出文件名
            output_filename = f"{output_prefix}.{drug_str}.time.h5ad"
            
            # 保存文件
            subset.write(output_filename)
            saved_files.append(output_filename)
            
            print(f"保存: {output_filename} (包含 {len(all_indices)} 个细胞，其中CTRL: {len(ctrl_indices)}, {drug_str}: {len(indices)})")
    
    # 生成汇总信息
    print(f"\n拆分完成! 共生成 {len(saved_files)} 个文件")
    print("生成的文件:")
    for filename in saved_files:
        print(f"  - {filename}")

def main():
    parser = argparse.ArgumentParser(description='根据h5ad文件中的gene列拆分成多个文件')
    parser.add_argument('input_file', help='输入的h5ad文件路径')
    parser.add_argument('output_prefix', help='输出文件前缀，格式为"前缀.药物.h5ad"')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件不存在: {args.input_file}")
        return
    
    try:
        split_h5ad_by_gene(args.input_file, args.output_prefix)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()
