#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

def analyze_convergence(iterations, losses, window_size=50, threshold=0.001):
    """
    分析loss是否还在下降
    
    参数:
    - iterations: 迭代次数列表
    - losses: 对应的loss值列表
    - window_size: 用于分析趋势的窗口大小
    - threshold: 判断收敛的阈值
    
    返回:
    - status: 收敛状态 ('下降', '平稳', '上升')
    - trend: 趋势值 (斜率)
    - recommendation: 训练建议
    """
    if len(losses) < window_size:
        # 如果数据点太少，使用所有数据
        window_size = len(losses)
    
    # 取最后window_size个点分析趋势
    recent_losses = losses[-window_size:]
    recent_iters = iterations[-window_size:]
    
    # 计算线性回归斜率
    if len(recent_losses) > 1:
        slope, _, r_value, _, _ = stats.linregress(range(len(recent_losses)), recent_losses)
        
        # 根据斜率判断趋势
        if slope < -threshold:
            status = "下降"
            recommendation = "继续训练"
        elif abs(slope) <= threshold:
            status = "平稳"
            recommendation = "接近收敛，可考虑停止"
        else:
            status = "上升"
            recommendation = "可能过拟合，建议检查"
        
        return status, slope, r_value**2, recommendation
    else:
        return "数据不足", 0, 0, "需要更多数据"

def main():
    if len(sys.argv) < 2:
        print("用法: python plot_loss.py [log文件1] [log文件2] ...")
        print("示例: python plot_loss.py *.log")
        sys.exit(1)
    
    # 创建DataFrame来存储所有数据
    all_data = {}
    convergence_analysis = {}
    
    print("正在处理文件:")
    # 处理每个log文件
    for file_path in sys.argv[1:]:
        filename = Path(file_path).stem  # 获取不带扩展名的文件名
        print(f"  {file_path} -> {filename}")
        
        iterations, losses = parse_log_file(file_path)
        
        if iterations:
            # 用文件名作为列名，iterations作为索引
            all_data[filename] = pd.Series(losses, index=iterations, name=filename)
            
            # 分析收敛状态
            status, slope, r_squared, recommendation = analyze_convergence(iterations, losses)
            convergence_analysis[filename] = {
                'status': status,
                'slope': slope,
                'r_squared': r_squared,
                'recommendation': recommendation,
                'final_loss': losses[-1],
                'total_iters': len(iterations)
            }
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
    
    # 打印收敛分析结果
    print("\n" + "="*80)
    print("收敛分析结果:")
    print("="*80)
    analysis_df = pd.DataFrame(convergence_analysis).T
    print(analysis_df[['status', 'slope', 'final_loss', 'recommendation']])
    
    # 打印需要继续训练的文件
    print("\n" + "="*80)
    print("建议继续训练的文件 (loss仍在下降):")
    print("="*80)
    continuing_files = [f for f, analysis in convergence_analysis.items() if analysis['status'] == '下降']
    if continuing_files:
        for file in continuing_files:
            print(f"  - {file}: 斜率 = {convergence_analysis[file]['slope']:.6f}, 最终loss = {convergence_analysis[file]['final_loss']:.4f}")
    else:
        print("  所有训练过程似乎都已收敛或接近收敛")
    
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

    # 保存收敛分析结果
    analysis_csv = 'convergence_analysis.csv'
    analysis_df.to_csv(analysis_csv)
    print(f"收敛分析结果已保存到: {analysis_csv}")
    
    plt.show()

if __name__ == "__main__":
    main()
