import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# 设置 SCI 一区风格：Times New Roman 字体，紧凑布局
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.3})

# 假设合并 CSV 数据（实际中从 CLUDA_MSL_F-5.csv 等加载并合并）
# 示例数据：Method, MSL_AUPRC, SMD_AUPRC, Boiler_AUPRC, Average_AUPRC
data = {
    'Method': ['DACAD', 'CLUDA', 'MSPAD'],
    'MSL': [0.88, 0.85, 0.92],
    'SMD': [0.84, 0.82, 0.88],
    'Boiler': [0.94, 0.92, 0.97],
    'Average': [0.89, 0.86, 0.92]
}
df = pd.DataFrame(data)

# 1. 柱状图：AUPRC 比较 (双栏友好)
fig, ax = plt.subplots(figsize=(8.5, 6))
datasets = ['MSL', 'SMD', 'Boiler', 'Average']
x = np.arange(len(datasets))
width = 0.25

for i, method in enumerate(df['Method']):
    ax.bar(x + i*width, df[method].values, width, label=method,
           color=['gray', 'orange', 'blue'][i], alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Datasets', fontsize=12, fontweight='bold')
ax.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
ax.set_title('Comparison of AUPRC across Methods', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(datasets)
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('comparison_auprc.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# 2. 折线图：MSPAD vs. DACAD 性能提升 %
df_elev = df.copy()
df_elev.iloc[:, 1:] = (df_elev.iloc[:, 1:] - df.iloc[0, 1:]) / df.iloc[0, 1:] * 100  # % 提升
df_elev = df_elev.iloc[[0, 2]]  # 只取 DACAD 和 MSPAD
df_elev = df_elev.melt(id_vars='Method', var_name='Dataset', value_name='Improvement %')

fig, ax = plt.subplots(figsize=(8.5, 6))
sns.lineplot(data=df_elev, x='Dataset', y='Improvement %', hue='Method', marker='o', linewidth=2.5,
             palette={'DACAD': 'gray', 'MSPAD': 'blue'}, ax=ax)
ax.set_xlabel('Datasets', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance Improvement (%)', fontsize=12, fontweight='bold')
ax.set_title('MSPAD Performance Gain over DACAD', fontsize=14, fontweight='bold', pad=20)
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('comparison_trend.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Generated PDFs: comparison_auprc.pdf, comparison_trend.pdf")
