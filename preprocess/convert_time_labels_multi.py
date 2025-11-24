#!/usr/bin/env python3
"""
将h5ad文件中的时间标签转换为浮点数格式
支持多文件合并
"""

import argparse
import anndata as ad
import pandas as pd
import numpy as np
import sys
import os
import warnings
from collections import Counter

def merge_h5ad_files(file_list, merge_method='inner'):
    """
    合并多个h5ad文件
    
    参数:
        file_list: h5ad文件路径列表
        merge_method: 合并方式 ('inner', 'outer', 'union')
    
    返回:
        合并后的AnnData对象
    """
    if not file_list:
        raise ValueError("文件列表为空")
    
    if len(file_list) == 1:
        print(f"单个文件模式: 读取 {file_list[0]}")
        return ad.read_h5ad(file_list[0])
    
    print(f"合并 {len(file_list)} 个h5ad文件...")
    print(f"文件列表: {file_list}")
    
    # 读取所有文件
    adatas = []
    for i, file_path in enumerate(file_list):
        print(f"读取文件 {i+1}/{len(file_list)}: {file_path}")
        adata = ad.read_h5ad(file_path)
        
        # 为每个文件添加来源标识（如果obs中还没有的话）
        if 'batch' not in adata.obs.columns:
            adata.obs['batch'] = f"batch_{i+1}"
        else:
            # 如果已有batch列，则追加文件标识
            adata.obs['batch'] = adata.obs['batch'].astype(str) + f"_file{i+1}"
        
        adatas.append(adata)
    
    # 检查变量（基因）的一致性
    var_intersection = set(adatas[0].var_names)
    var_union = set(adatas[0].var_names)
    
    for i, adata in enumerate(adatas[1:], 1):
        var_intersection = var_intersection.intersection(set(adata.var_names))
        var_union = var_union.union(set(adata.var_names))
    
    print(f"变量交集数量: {len(var_intersection)}")
    print(f"变量并集数量: {len(var_union)}")
    
    # 根据合并方式处理
    if merge_method == 'inner':
        print("使用inner合并方式（保留共有变量）")
        # 保留所有文件中都存在的变量
        common_vars = list(var_intersection)
        if not common_vars:
            raise ValueError("错误: 文件之间没有共同的变量（基因）")
        
        # 过滤每个adata只保留共有变量
        for i in range(len(adatas)):
            adatas[i] = adatas[i][:, common_vars].copy()
        
        # 合并数据
        merged_adata = ad.concat(adatas, axis=0, join='inner', index_unique=None)
        
    elif merge_method == 'outer':
        print("使用outer合并方式（保留所有变量，缺失值用0填充）")
        merged_adata = ad.concat(adatas, axis=0, join='outer', index_unique=None)
        # 填充缺失值为0
        merged_adata.X = merged_adata.X.toarray() if hasattr(merged_adata.X, 'toarray') else merged_adata.X
        merged_adata.X = np.nan_to_num(merged_adata.X, nan=0.0)
        
    elif merge_method == 'union':
        print("使用union合并方式（保留所有变量，使用稀疏矩阵）")
        merged_adata = ad.concat(adatas, axis=0, join='outer', index_unique=None)
    else:
        raise ValueError(f"不支持的合并方式: {merge_method}")
    
    print(f"合并完成: {merged_adata.shape[0]} 个细胞, {merged_adata.shape[1]} 个变量")
    print(f"批次分布: {dict(Counter(merged_adata.obs['batch']))}")
    
    return merged_adata

def convert_time_labels(h5ad_files, time_obs_label, original_labels, converted_labels, output_file, merge_method='inner'):
    """
    转换h5ad文件中的时间标签为浮点数
    
    参数:
        h5ad_files: 输入的h5ad文件路径列表
        time_obs_label: 包含时间信息的obs列名
        original_labels: 原始标签列表
        converted_labels: 转换后的标签列表
        output_file: 输出文件路径
        merge_method: 文件合并方式
    """
    
    try:
        # 检查所有输入文件是否存在
        for file_path in h5ad_files:
            if not os.path.exists(file_path):
                raise ValueError(f"错误: 输入文件 '{file_path}' 不存在")
        
        # 合并或读取文件
        if len(h5ad_files) > 1:
            adata = merge_h5ad_files(h5ad_files, merge_method)
        else:
            print(f"正在读取文件: {h5ad_files[0]}")
            adata = ad.read_h5ad(h5ad_files[0])
        
        # 检查时间标签列是否存在
        if time_obs_label not in adata.obs.columns:
            available_columns = list(adata.obs.columns)
            raise ValueError(f"错误: obs中找不到标签 '{time_obs_label}'\n可用的列有: {available_columns}")
        
        # 创建映射字典
        if len(original_labels) != len(converted_labels):
            raise ValueError("错误: 原始标签和转换后标签数量不一致")
        
        label_map = {}
        for orig, conv in zip(original_labels, converted_labels):
            try:
                # 尝试将转换后的标签转换为浮点数
                float_conv = float(conv)
                label_map[orig] = float_conv
            except ValueError:
                raise ValueError(f"错误: 转换标签 '{conv}' 无法转换为浮点数")
        
        print(f"标签映射关系: {label_map}")
        
        # 检查原始数据中是否有未映射的标签
        unique_labels = set(adata.obs[time_obs_label].astype(str))
        unmapped_labels = unique_labels - set(label_map.keys())
        
        if unmapped_labels:
            print(f"警告: 发现未映射的标签: {unmapped_labels}")
            print("这些标签将被设置为NaN")
        
        # 转换时间标签
        def map_time_label(label):
            str_label = str(label)
            return label_map.get(str_label, np.nan)
        
        # 应用映射
        adata.obs['time'] = adata.obs[time_obs_label].apply(map_time_label)
        
        # 检查转换结果
        time_stats = adata.obs['time'].describe()
        print("\n时间列统计信息:")
        print(f"非空值数量: {adata.obs['time'].notna().sum()}")
        print(f"唯一值: {sorted(adata.obs['time'].dropna().unique())}")
        print(f"数据类型: {adata.obs['time'].dtype}")
        
        # 显示批次信息（如果有多文件）
        if len(h5ad_files) > 1:
            print(f"\n批次信息:")
            batch_time_stats = adata.obs.groupby('batch')['time'].apply(lambda x: f"非空: {x.notna().sum()}, 唯一值: {sorted(x.dropna().unique())}")
            for batch, stats in batch_time_stats.items():
                print(f"  {batch}: {stats}")
        
        # 保存结果
        print(f"\n正在保存到: {output_file}")
        adata.write_h5ad(output_file)
        print("转换完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='将h5ad文件中的时间标签转换为浮点数格式（支持多文件合并）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 单文件转换
  python convert_time_labels.py \\
    --input data.h5ad \\
    --time-label stage \\
    --original "E10.5,E11.5,E12.5" \\
    --converted "10.5,11.5,12.5" \\
    --output data_with_time.h5ad
  
  # 多文件合并转换
  python convert_time_labels.py \\
    --input "file1.h5ad,file2.h5ad,file3.h5ad" \\
    --time-label stage \\
    --original "E10.5,E11.5,E12.5" \\
    --converted "10.5,11.5,12.5" \\
    --merge-method inner \\
    --output merged_data_with_time.h5ad
    
  # 使用简写参数
  python convert_time_labels.py \\
    -i "sample1.h5ad,sample2.h5ad" \\
    -t time_point \\
    -o "day0,day1,day2,day3" \\
    -c "0,1,2,3" \\
    --output converted_merged.h5ad
        '''
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='输入的h5ad文件路径，多个文件用逗号分隔')
    parser.add_argument('--time-label', '-t', required=True,
                       help='包含时间信息的obs列名')
    parser.add_argument('--original', '-o', required=True,
                       help='原始标签列表，逗号分隔')
    parser.add_argument('--converted', '-c', required=True,
                       help='转换后的浮点数标签列表，逗号分隔')
    parser.add_argument('--merge-method', choices=['inner', 'outer', 'union'], 
                       default='inner',
                       help='多文件合并方式: inner(交集), outer(并集填充0), union(并集稀疏矩阵)')
    parser.add_argument('--output', required=True,
                       help='输出的h5ad文件路径')
    
    args = parser.parse_args()
    
    # 解析输入文件列表
    input_files = [file_path.strip() for file_path in args.input.split(',')]
    
    # 解析标签列表
    original_labels = [label.strip() for label in args.original.split(',')]
    converted_labels = [label.strip() for label in args.converted.split(',')]
    
    print("=" * 60)
    print("h5ad时间标签转换工具 (支持多文件合并)")
    print("=" * 60)
    print(f"输入文件: {input_files}")
    print(f"文件数量: {len(input_files)}")
    print(f"时间标签列: {args.time_label}")
    print(f"原始标签: {original_labels}")
    print(f"转换标签: {converted_labels}")
    print(f"合并方式: {args.merge_method}")
    print(f"输出文件: {args.output}")
    print("=" * 60)
    
    # 执行转换
    convert_time_labels(
        h5ad_files=input_files,
        time_obs_label=args.time_label,
        original_labels=original_labels,
        converted_labels=converted_labels,
        output_file=args.output,
        merge_method=args.merge_method
    )

if __name__ == "__main__":
    main()
