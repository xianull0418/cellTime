import scanpy as sc
import pandas as pd
import plotly.graph_objects as go
import re
import sys

# -----------------------------
# 1. 读取 h5ad 文件
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python sankey_lineage_plotly.py <input.h5ad>")
    sys.exit(1)

h5ad_file = sys.argv[1]
print(f"Loading data from {h5ad_file}...")
adata = sc.read_h5ad(h5ad_file)

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
# 2. 构建时间点之间的 lineage 流
# -----------------------------
timepoints = sorted(obs['time_numeric'].dropna().unique())
if len(timepoints) < 2:
    raise ValueError("Need at least two time points to draw Sankey diagram.")

# Count cells per timepoint
cell_counts = obs.groupby('time_numeric').size().to_dict()

# Build flows: from t_i to t_{i+1}
source = []
target = []
value = []
label = []

# 创建所有节点：每个时间点一个节点
node_labels = [f"Time {t}" for t in timepoints]
node_ids = {t: i for i, t in enumerate(timepoints)}

# 添加从每个时间点到下一个时间点的 flow
for i in range(len(timepoints) - 1):
    t_curr = timepoints[i]
    t_next = timepoints[i+1]
    
    # Cells at t_curr that have a next_cell_id != -1
    mask = (obs['time_numeric'] == t_curr) & (obs['next_cell_id'] != -1)
    flow_count = mask.sum()
    
    if flow_count > 0:
        source.append(node_ids[t_curr])
        target.append(node_ids[t_next])
        value.append(flow_count)

# -----------------------------
# 3. 绘制 Sankey 图（Plotly）
# -----------------------------
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=node_labels,
        color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0", "#ffb3e6"] * 10  # 循环颜色
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=['rgba(255,153,153,0.5)', 'rgba(102,179,255,0.5)', 'rgba(153,255,153,0.5)',
               'rgba(255,204,153,0.5)', 'rgba(194,194,240,0.5)', 'rgba(255,179,230,0.5)'] * 10
    )
)])

fig.update_layout(
    title_text="Cell Lineage Sankey Diagram Over Time",
    font_size=12,
    width=1000,
    height=600,
    margin=dict(l=20, r=20, t=50, b=20),
    paper_bgcolor='white'
)

# -----------------------------
# 4. 保存为 PDF（需要安装 kaleido）
# -----------------------------
try:
    fig.write_image("cell_lineage_sankey.pdf", format="pdf", width=1000, height=600)
    print("Sankey diagram saved as 'cell_lineage_sankey.pdf'")
except Exception as e:
    print(f"⚠️ 保存 PDF 失败：{e}")
    print("请先安装 kaleido: pip install kaleido")
    print("然后重试。或者手动在浏览器中打开交互式图并截图。")

# 显示图（可选）
fig.show()
