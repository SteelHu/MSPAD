import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams
from math import radians
import matplotlib.patches as mpatches

# 设置 SCI 一区风格
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
sns.set_style("white")

# 假设加载 CSV（实际从 Sensitivity_MSL_F-5_C-1.csv 等合并）
# 示例数据：param_name, param_value, avg_prc (Average AUPRC)
data = {
    'param_name': ['weight_ms_disc']*7 + ['prototypical_margin']*8 + ['scale_weights']*5,
    'param_value': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7,
                    0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5,
                    '[0.1,0.3,0.6]', '[0.33,0.33,0.34]', '[0.6,0.3,0.1]', '[0.0,0.0,1.0]', '[0.2,0.4,0.4]'],
    'AUPRC': [0.88, 0.89, 0.90, 0.92, 0.91, 0.90, 0.89,  # 示例值
              0.89, 0.90, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87,
              0.92, 0.90, 0.89, 0.91, 0.91]
}
df = pd.DataFrame(data)

# 1. 折线图：AUPRC vs. 参数值 (多参数)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 6))
# weight_ms_disc
df_w = df[df['param_name'] == 'weight_ms_disc']
sns.lineplot(data=df_w, x='param_value', y='AUPRC', marker='o', linewidth=2.5, ax=ax1,
             color='blue', markersize=6)
ax1.set_xlabel('weight_loss_ms_disc', fontsize=12, fontweight='bold')
ax1.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
ax1.set_title('Sensitivity to Multi-Scale Weight', fontsize=12, fontweight='bold', pad=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# prototypical_margin
df_m = df[df['param_name'] == 'prototypical_margin']
sns.lineplot(data=df_m, x='param_value', y='AUPRC', marker='s', linewidth=2.5, ax=ax2,
             color='green', markersize=6)
ax2.set_xlabel('prototypical_margin', fontsize=12, fontweight='bold')
ax2.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
ax2.set_title('Sensitivity to Margin', fontsize=12, fontweight='bold', pad=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('sensitivity_lines.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# 2. 雷达图：多配置性能 (简化 scale_weights)
categories = ['AUPRC', 'F1', 'AUC']  # 示例指标
configs = ['[0.1,0.3,0.6]', 'Uniform', 'Reverse', 'High Only', 'Mid-High']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(8.5, 6), subplot_kw=dict(projection='polar'))
for i, config in enumerate(configs):
    # 示例值：每个配置的 [AUPRC, F1, AUC] (0-1 归一化)
    values = [0.92, 0.87, 0.95] if config == '[0.1,0.3,0.6]' else [0.90, 0.85, 0.93]  # 简化
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=config, markersize=6)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Scale Weights Configuration Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True, fancybox=True, shadow=True)
ax.grid(True)
plt.tight_layout()
plt.savefig('sensitivity_radar.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated PDFs: sensitivity_lines.pdf, sensitivity_radar.pdf")
