import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import numpy as np
import sys
import re

# -----------------------------
# 1. 读取 h5ad 文件
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python sankey_lineage.py <input.h5ad>")
    sys.exit(1)

h5ad_file = sys.argv[1]
print(f"Loading data from {h5ad_file}...")
adata = sc.read_h5ad(h5ad_file)

# -----------------------------
# 2. 提取并预处理 obs
# -----------------------------
obs = adata.obs.copy()

# 确保必要列存在
required_cols = ['time', 'prev_cell_id', 'next_cell_id']
for col in required_cols:
    if col not in obs.columns:
        raise ValueError(f"Column '{col}' not found in adata.obs")

# 处理 time 列：如果是字符串如 "E3", "E5.5"，尝试提取数字
def extract_time_numeric(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    # Try to extract number from string like "E3", "E5.5", "E7"
    match = re.search(r'[\d.]+', str(val))
    if match:
        return float(match.group())
    else:
        raise ValueError(f"Cannot parse time from value: {val}")

obs['time_numeric'] = obs['time'].apply(extract_time_numeric)

# 转换 prev/next to int (ensure -1 is integer)
obs['prev_cell_id'] = pd.to_numeric(obs['prev_cell_id'], errors='coerce').fillna(-1).astype(int)
obs['next_cell_id'] = pd.to_numeric(obs['next_cell_id'], errors='coerce').fillna(-1).astype(int)

# -----------------------------
# 3. 构建时间点之间的 lineage 流
# -----------------------------
timepoints = sorted(obs['time_numeric'].dropna().unique())
if len(timepoints) < 2:
    raise ValueError("Need at least two time points to draw Sankey diagram.")

# Count cells per timepoint
cell_counts = obs.groupby('time_numeric').size()

# Build flows: from t_i to t_{i+1}
flows = []  # (from_time, to_time, flow_count)
for i in range(len(timepoints) - 1):
    t_curr = timepoints[i]
    t_next = timepoints[i+1]
    
    # Cells at t_curr that have a next_cell_id != -1
    # Note: We don't validate that next_cell_id actually belongs to t_next
    # (assuming data is consistent)
    mask = (obs['time_numeric'] == t_curr) & (obs['next_cell_id'] != -1)
    flow_count = mask.sum()
    flows.append((t_curr, t_next, flow_count))

# -----------------------------
# 4. 绘制 Sankey 图
# -----------------------------
labels = [f"Time {t}" for t in timepoints]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])

sankey = Sankey(ax=ax, scale=0.01, offset=0.3, margin=0.4)

# First stage: all cells at timepoints[0]
initial_count = cell_counts[timepoints[0]]
first_ending = initial_count - flows[0][2]

sankey.add(
    flows=[initial_count, -flows[0][2], -first_ending],
    labels=['', labels[1], 'End'],
    orientations=[0, 0, -1],
    pathlengths=[0.25, 0.25, 0.25]
)

# Middle stages
for i in range(1, len(timepoints) - 1):
    t_curr = timepoints[i]
    total_here = cell_counts[t_curr]
    continuing = flows[i][2] if i < len(flows) else 0
    ending_here = total_here - continuing
    
    sankey.add(
        flows=[flows[i-1][2], -continuing, -ending_here],
        labels=['', labels[i+1] if i+1 < len(labels) else '', 'End'],
        orientations=[0, 0, -1],
        prior=i-1,
        connect=(1, 0),
        pathlengths=[0.25, 0.25, 0.25]
    )

# Final timepoint: all cells end (no next)
if len(timepoints) >= 2:
    last_t = timepoints[-1]
    last_total = cell_counts[last_t]
    # All cells at last timepoint end
    sankey.add(
        flows=[flows[-1][2], -last_total],
        labels=['', 'End'],
        orientations=[0, -1],
        prior=len(timepoints)-2,
        connect=(1, 0),
        pathlengths=[0.25, 0.25]
    )

diagrams = sankey.finish()
plt.title("Cell Lineage Sankey Diagram Over Time", fontsize=14)
plt.tight_layout()

# -----------------------------
# 5. 保存为 PDF
# -----------------------------
output_pdf = "cell_lineage_sankey.pdf"
plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
print(f"Sankey diagram saved as '{output_pdf}'")

# Optional: show plot
# plt.show()
