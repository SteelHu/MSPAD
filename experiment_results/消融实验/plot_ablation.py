import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# 设置 SCI 一区风格
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
sns.set_style("white")

# 假设加载 CSV（实际从 ablation_results_MSL_F-5_C-1.csv 等合并，聚焦 Average AUPRC）
# 示例数据：简化 25 配置为组 (Core, Layers, Losses)
data = {
    'Group': ['Core', 'Core', 'Core', 'Core', 'Core',  # Abl-4.x
              'Layers', 'Layers', 'Layers', 'Layers', 'Layers', 'Layers', 'Layers',  # Abl-5.x
              'Losses', 'Losses', 'Losses', 'Losses', 'Losses', 'Losses'],  # Abl-6.x
    'Config': ['Baseline', '+Multi-Scale', '+Prototypical', '+Weighted', 'Full',
               'L1 Only', 'L2 Only', 'L3 Only', 'L1+2', 'L2+3', 'L1+3', 'All',
               'Full', 'w/o Single DA', 'w/o Multi DA', 'w/o Proto', 'w/o Src CL', 'w/o Trg CL'],
    'AUPRC': [0.89, 0.90, 0.90, 0.91, 0.92,  # 示例值
              0.87, 0.89, 0.90, 0.89, 0.91, 0.90, 0.92,
              0.92, 0.90, 0.89, 0.86, 0.91, 0.91]
}
df = pd.DataFrame(data)

# 1. 分组柱状图：AUPRC 按组
fig, ax = plt.subplots(figsize=(8.5, 6))
groups = df.groupby('Group')
for name, group in groups:
    x_pos = range(len(group))
    ax.bar(x_pos, group['AUPRC'], label=name, alpha=0.8, edgecolor='black', linewidth=0.5,
           color=['skyblue', 'lightcoral', 'lightgreen'][list(groups.keys()).index(name)])

ax.set_xlabel('Configurations', fontsize=12, fontweight='bold')
ax.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study: AUPRC by Component Groups', fontsize=14, fontweight='bold', pad=20)
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)
# 旋转 x 标签以适应双栏
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('ablation_bars.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# 2. 热图：层组合性能 (简化 Abl-5.x 为矩阵)
layer_combs = ['L1 Only', 'L2 Only', 'L3 Only', 'L1+2', 'L2+3', 'L1+3', 'All']
auprc_matrix = np.array([[0.87, 0.89, 0.90, 0.89, 0.91, 0.90, 0.92]])  # 示例行 (Average)
df_heat = pd.DataFrame(auprc_matrix, columns=layer_combs, index=['Average'])

plt.figure(figsize=(8.5, 2))  # 双栏宽度，短高度
sns.heatmap(df_heat, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'AUPRC'},
            linewidths=0.5, linecolor='white', ax=plt.gca())
plt.title('Layer Combination Ablation: AUPRC Heatmap', fontsize=12, fontweight='bold', pad=20)
plt.xlabel('Layer Combinations', fontsize=10)
plt.ylabel('Dataset Average', fontsize=10)
plt.tight_layout()
plt.savefig('ablation_heatmap.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated PDFs: ablation_bars.pdf, ablation_heatmap.pdf")
