#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

def parse_log_file(file_path):
    """解析log文件，提取Iter和loss"""
    iterations = []
    losses = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 匹配格式: Iter: X, loss: Y
            match = re.search(r'Iter:\s*(\d+),\s*loss:\s*([\d.]+)', line)
            if match:
                iter_num = int(match.group(1))
                loss_val = float(match.group(2))
                iterations.append(iter_num)
                losses.append(loss_val)
    
    return iterations, losses

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_loss.py [log文件1] [log文件2] ...")
        print("示例: python plot_loss.py *.log")
        sys.exit(1)
    
    # 创建DataFrame来存储所有数据
    all_data = {}
    
    print("正在处理文件:")
    # 处理每个log文件
    for file_path in sys.argv[1:]:
        filename = Path(file_path).stem  # 获取不带扩展名的文件名
        print(f"  {file_path} -> {filename}")
        
        iterations, losses = parse_log_file(file_path)
        
        if iterations:
            # 用文件名作为列名，iterations作为索引
            all_data[filename] = pd.Series(losses, index=iterations, name=filename)
        else:
            print(f"  警告: 在 {file_path} 中未找到Iter和loss数据")
    
    if not all_data:
        print("没有找到有效的Iter和loss数据")
        sys.exit(1)
    
    # 创建DataFrame
    df = pd.DataFrame(all_data)
    df.index.name = 'Iter'
    
    # 按Iter排序
    df = df.sort_index()
    
    # 保存为CSV文件
    csv_filename = 'loss_data.csv'
    df.to_csv(csv_filename)
    print(f"\n数据已保存到: {csv_filename}")
    
    # 打印数据概览
    print("\n数据概览:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 绘制loss曲线
    plt.figure(figsize=(12, 8))
    
    # 为不同的列使用不同的颜色和样式
    colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns)))
    linestyles = ['-', '--', '-.', ':'] * 3
    
    for i, column in enumerate(df.columns):
        plt.plot(df.index, df[column], 
                label=column, 
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2,
                marker='o' if len(df) < 20 else '',  # 数据点少时显示标记点
                markersize=4)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 修复警告1: 检查是否有可能除以零的情况
    min_val = df.min().min()
    max_val = df.max().max()
    
    # 只有当最小值大于0时才考虑对数坐标
    if True: #min_val > 0 and max_val / min_val > 1000:
        plt.yscale('log')
        plt.ylabel('Loss (log scale)')
        print("使用对数坐标轴")
    
    # 修复警告2: 更好的图例布局处理
    # 先尝试tight_layout，如果失败则使用subplots_adjust
    try:
        plt.tight_layout()
    except:
        print("警告: tight_layout失败，使用手动调整")
        plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
    
    # 添加图例（在调整布局后）
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # 保存图片
    plot_filename = 'loss_curves.pdf'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {plot_filename}")
    
    plt.show()

if __name__ == "__main__":
    main()
