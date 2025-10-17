import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# --- 步骤 1: 准备数据 ---
data = {
    'Methods': ['YOLOv8-IR', 'YOLOv8-RGB', 'CFT [53]', 'MFPT [58]', 'CrossFormer [50]', 'Fusion-Mamba [57]',
                'DHAF-Net (ours)'],
    'Precision': [75.1, 71.6, 77.6, 79.3, 78.1, 83.4, 81.7],
    'Recall': [65.3, 62.2, 72.3, 72.9, 72.8, 77.1, 76.4],
    'F1': [69.9, 66.6, 74.9, 76.0, 75.4, 80.1, 78.7],
    'AP50': [72.9, 66.3, 78.7, 80.0, 79.3, 84.9, 82.1],
    'mAP': [38.3, 28.2, 40.2, 41.9, 42.1, 47.0, 48.1],
    'Param. (M)': [76.7, 76.7, 206.0, 200.0, 340.0, 287.6, 124.9],
    'Time (ms)': [22, 22, 68, 80, 80, 78, 49]
}
df = pd.DataFrame(data)

# --- 步骤 2: 数据转换与归一化 ---
ranges_and_ticks = {
    'Precision': {'range': [70, 85], 'ticks': [70, 73, 76, 79, 82, 85]},
    'Recall': {'range': [60, 80], 'ticks': [60, 64, 68, 72, 76, 80]},
    'F1': {'range': [65, 85], 'ticks': [65, 69, 73, 77, 81, 85]},
    'AP50': {'range': [65, 90], 'ticks': [65, 70, 75, 80, 85, 90]},
    'mAP': {'range': [25, 50], 'ticks': [25, 30, 35, 40, 45, 50]},
    'Param. (M)': {'range': [70, 370], 'ticks': [70, 130, 190, 250, 310, 370]},
    'Time (ms)': {'range': [20, 95], 'ticks': [20, 35, 50, 65, 80, 95]}
}
cost_metrics = ['Param. (M)', 'Time (ms)']
df_normalized = df.copy()
for col, props in ranges_and_ticks.items():
    min_val, max_val = props['range']
    if col in cost_metrics:
        df_normalized[col] = (max_val - df[col]) / (max_val - min_val)
    else:
        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)

labels = list(ranges_and_ticks.keys())
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# --- 步骤 3: 绘图流程 ---
TITLE_SIZE = 30
LABEL_SIZE = 30
LEGEND_SIZE = 26
TICK_SIZE = 24

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw=dict(polar=True))

ax.grid(False)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_frame_on(False)

# --- 绘制背景和轴线 ---
radial_ticks_normalized = np.linspace(0, 1, 6)
full_circle_angles = np.linspace(0, 2 * np.pi, 200)

# 绘制交替颜色的背景填充
for i in range(len(radial_ticks_normalized) - 1, 0, -1):
    if i % 2 == 1:
        ax.fill_between(full_circle_angles, radial_ticks_normalized[i - 1], radial_ticks_normalized[i], color='#f0f0f0',
                        zorder=0)

for i in range(num_vars):
    angle = angles[i]
    ax.plot([angle, angle], [0, 1.0], color='grey', linestyle='--', linewidth=1.5, zorder=1)
ax.plot(full_circle_angles, [1.0] * len(full_circle_angles), color='grey', linewidth=2.5, zorder=1)

# --- 绘制刻度和指标名称 ---
for i, (label, props) in enumerate(ranges_and_ticks.items()):
    angle = angles[i]
    ticks = props['ticks']
    min_val, max_val = props['range']
    angle_deg = np.rad2deg(angle)

    if angle_deg == 0:
        ha, va = 'left', 'center'
    elif 0 < angle_deg < 90:
        ha, va = 'left', 'bottom'
    elif angle_deg == 90:
        ha, va = 'center', 'bottom'
    elif 90 < angle_deg < 180:
        ha, va = 'right', 'bottom'
    elif angle_deg == 180:
        ha, va = 'right', 'center'
    elif 180 < angle_deg < 270:
        ha, va = 'right', 'top'
    elif angle_deg == 270:
        ha, va = 'center', 'top'
    else:
        ha, va = 'left', 'top'

    # <<< 修改点：将 1.12 修改为 1.08，使指标名称更靠近图表 >>>
    ax.text(angle, 1.08, label, ha=ha, va=va, fontweight='bold', fontsize=LABEL_SIZE)

    for tick_val in ticks:
        if label in cost_metrics:
            radial_pos = (max_val - tick_val) / (max_val - min_val)
        else:
            radial_pos = (tick_val - min_val) / (max_val - min_val)

        if abs(radial_pos) > 0.01:
            ax.text(angle, radial_pos, f"{int(tick_val)}",
                    ha='center', va='center', fontsize=TICK_SIZE, color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='none', alpha=0.8))

# --- 绘制数据线 ---
vibrant_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#e41a1c']
method_colors = {}
our_method_color = vibrant_colors.pop()
color_idx = 0
for method in df['Methods']:
    if method == 'DHAF-Net (ours)':
        method_colors[method] = our_method_color
    else:
        method_colors[method] = vibrant_colors[color_idx]
        color_idx += 1

for i, row in df.iterrows():
    method_name = row['Methods']
    values = df_normalized.loc[i, labels].values.flatten().tolist()
    values += values[:1]
    linewidth = 4 if method_name == 'DHAF-Net (ours)' else 4
    zorder = 10 if method_name == 'DHAF-Net (ours)' else 10
    ax.plot(angles, values, color=method_colors[method_name], linewidth=linewidth,
            linestyle='solid', label=method_name, zorder=zorder)

# --- 步骤 5: 美化与收尾 ---
# 调整Y轴范围，为标题留出空间，新的标签位置(1.08)也需要考虑
ax.set_ylim(0, 1.3)

legend = ax.legend(
    title='Methods',
    title_fontsize=LEGEND_SIZE + 2,
    loc='upper right',
    bbox_to_anchor=(1.28, 1.15),
    fontsize=LEGEND_SIZE,
    frameon=True,
    shadow=True
)

# 调整标题的 y 位置，以适应更紧凑的布局
ax.set_title('Comprehensive Performance Comparison of Models', size=TITLE_SIZE, weight='bold', y=1.1)

# 调整布局以防止图例被截断
fig.subplots_adjust(right=0.8, top=0.9)  # 同时也调整了top，为标题留出空间

plt.show()