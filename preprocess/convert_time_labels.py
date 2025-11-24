#!/usr/bin/env python3
"""
将h5ad文件中的时间标签转换为浮点数格式
"""

import argparse
import anndata as ad
import pandas as pd
import numpy as np
import sys
import os

def convert_time_labels(h5ad_file, time_obs_label, original_labels, converted_labels, output_file):
    """
    转换h5ad文件中的时间标签为浮点数
    
    参数:
        h5ad_file: 输入的h5ad文件路径
        time_obs_label: 包含时间信息的obs列名
        original_labels: 原始标签列表
        converted_labels: 转换后的标签列表
        output_file: 输出文件路径
    """
    
    try:
        # 读取h5ad文件
        print(f"正在读取文件: {h5ad_file}")
        adata = ad.read_h5ad(h5ad_file)
        
        # 检查时间标签列是否存在
        if time_obs_label not in adata.obs.columns:
            raise ValueError(f"错误: obs中找不到标签 '{time_obs_label}'")
        
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
        print(f"唯一值: {adata.obs['time'].dropna().unique()}")
        print(f"数据类型: {adata.obs['time'].dtype}")
        
        # 保存结果
        print(f"\n正在保存到: {output_file}")
        adata.write_h5ad(output_file)
        print("转换完成!")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='将h5ad文件中的时间标签转换为浮点数格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  python convert_time_labels.py \\
    --input data.h5ad \\
    --time-label stage \\
    --original "E10.5,E11.5,E12.5" \\
    --converted "10.5,11.5,12.5" \\
    --output data_with_time.h5ad
    
  python convert_time_labels.py \\
    -i sample.h5ad \\
    -t time_point \\
    -o "day0,day1,day2,day3" \\
    -c "0,1,2,3" \\
    --output converted_sample.h5ad
        '''
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='输入的h5ad文件路径')
    parser.add_argument('--time-label', '-t', required=True,
                       help='包含时间信息的obs列名')
    parser.add_argument('--original', '-o', required=True,
                       help='原始标签列表，逗号分隔')
    parser.add_argument('--converted', '-c', required=True,
                       help='转换后的浮点数标签列表，逗号分隔')
    parser.add_argument('--output', required=True,
                       help='输出的h5ad文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在")
        sys.exit(1)
    
    # 解析标签列表
    original_labels = [label.strip() for label in args.original.split(',')]
    converted_labels = [label.strip() for label in args.converted.split(',')]
    
    print("=" * 50)
    print("h5ad时间标签转换工具")
    print("=" * 50)
    print(f"输入文件: {args.input}")
    print(f"时间标签列: {args.time_label}")
    print(f"原始标签: {original_labels}")
    print(f"转换标签: {converted_labels}")
    print(f"输出文件: {args.output}")
    print("=" * 50)
    
    # 执行转换
    convert_time_labels(
        h5ad_file=args.input,
        time_obs_label=args.time_label,
        original_labels=original_labels,
        converted_labels=converted_labels,
        output_file=args.output
    )

if __name__ == "__main__":
    main()
