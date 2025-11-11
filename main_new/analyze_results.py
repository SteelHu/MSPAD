"""
实验结果分析和可视化脚本
==========================
功能：从实验结果中提取指标并生成可视化图表

使用方法：
python main_new/analyze_results.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from glob import glob
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def extract_metrics_from_log(log_file):
    """从日志文件中提取评估指标"""
    metrics = {}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取AUPRC
            auprc_match = re.search(r'AUPRC score is\s*:\s*([\d.]+)', content)
            if auprc_match:
                metrics['AUPRC'] = float(auprc_match.group(1))
            
            # 提取F1 Score
            f1_match = re.search(r'Best F1 score is\s*:\s*([\d.]+)', content)
            if f1_match:
                metrics['F1'] = float(f1_match.group(1))
            
            # 提取Precision
            prec_match = re.search(r'Best Prec score is\s*:\s*([\d.]+)', content)
            if prec_match:
                metrics['Precision'] = float(prec_match.group(1))
            
            # 提取Recall
            rec_match = re.search(r'Best Rec score is\s*:\s*([\d.]+)', content)
            if rec_match:
                metrics['Recall'] = float(rec_match.group(1))
            
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics


def collect_ablation_results():
    """收集消融实验结果"""
    results = []
    
    # 实验文件夹映射
    experiment_mapping = {
        'MSL_Baseline': 'Baseline',
        'MSL_MSDA_Full': 'Exp-1.1 (Full)',
        'MSL_MSDA_MSOnly': 'Exp-1.2 (MS Only)',
        'MSL_MSDA_Layer1Only': 'Exp-2.1 (Layer 1)',
        'MSL_MSDA_Layer2Only': 'Exp-2.2 (Layer 2)',
        'MSL_MSDA_Layer3Only': 'Exp-2.3 (Layer 3)',
        'MSL_MSDA_AllLayers': 'Exp-2.7 (All Layers)',
        'MSL_MSDA_Weights_Default': 'Exp-3.1 (Default Weights)',
        'MSL_MSDA_Weights_Uniform': 'Exp-3.2 (Uniform Weights)',
        'MSL_MSDA_Weights_Reverse': 'Exp-3.3 (Reverse Weights)',
    }
    
    for folder, exp_name in experiment_mapping.items():
        log_files = glob(f'results/{folder}/F-5-C-1/eval_*.log')
        if log_files:
            metrics = extract_metrics_from_log(log_files[0])
            if metrics:
                metrics['Experiment'] = exp_name
                metrics['Folder'] = folder
                results.append(metrics)
        else:
            print(f"⚠ No log file found for {exp_name} ({folder})")
    
    return pd.DataFrame(results)


def collect_sensitivity_results(param_name='weight_loss_ms_disc'):
    """收集参数敏感性分析结果"""
    results = []
    
    # 查找所有敏感性分析结果
    pattern = f'results/MSL_Sensitivity_{param_name}_*/F-5-C-1/eval_*.log'
    log_files = glob(pattern)
    
    for log_file in log_files:
        # 从路径中提取参数值
        match = re.search(rf'{param_name}_([\d.]+)', log_file)
        if match:
            param_value = float(match.group(1))
            metrics = extract_metrics_from_log(log_file)
            if metrics:
                metrics[param_name] = param_value
                results.append(metrics)
    
    return pd.DataFrame(results)


def generate_ablation_report(df):
    """生成消融实验报告"""
    if df.empty:
        print("No ablation results found!")
        return
    
    print("\n" + "="*80)
    print("Ablation Experiment Results")
    print("="*80)
    
    # 按AUPRC排序
    df_sorted = df.sort_values('AUPRC', ascending=False)
    
    print("\nResults (sorted by AUPRC):")
    print("-"*80)
    print(df_sorted[['Experiment', 'AUPRC', 'F1', 'Precision', 'Recall']].to_string(index=False))
    
    # 保存到CSV
    output_file = 'ablation_results_analysis.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # 计算改进
    if 'Baseline' in df_sorted['Experiment'].values:
        baseline_auprc = df_sorted[df_sorted['Experiment'] == 'Baseline']['AUPRC'].values[0]
        print(f"\nBaseline AUPRC: {baseline_auprc:.4f}")
        print("\nImprovements over Baseline:")
        print("-"*80)
        for _, row in df_sorted.iterrows():
            if row['Experiment'] != 'Baseline':
                improvement = row['AUPRC'] - baseline_auprc
                improvement_pct = (improvement / baseline_auprc) * 100
                print(f"{row['Experiment']}: {improvement:+.4f} ({improvement_pct:+.2f}%)")


def generate_sensitivity_report(df, param_name='weight_loss_ms_disc'):
    """生成敏感性分析报告"""
    if df.empty:
        print(f"No sensitivity results found for {param_name}!")
        return
    
    print("\n" + "="*80)
    print(f"Sensitivity Analysis Results: {param_name}")
    print("="*80)
    
    # 按参数值排序
    df_sorted = df.sort_values(param_name)
    
    print("\nResults:")
    print("-"*80)
    print(df_sorted[[param_name, 'AUPRC', 'F1', 'Precision', 'Recall']].to_string(index=False))
    
    # 保存到CSV
    output_file = f'sensitivity_analysis_{param_name}.csv'
    df_sorted.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # 找到最优值
    best_idx = df_sorted['AUPRC'].idxmax()
    best_row = df_sorted.loc[best_idx]
    
    print(f"\nBest Configuration:")
    print(f"  {param_name} = {best_row[param_name]}")
    print(f"  AUPRC = {best_row['AUPRC']:.4f}")
    print(f"  F1 = {best_row['F1']:.4f}")


def plot_sensitivity_analysis(df, param_name='weight_loss_ms_disc', save_plot=True):
    """绘制参数敏感性分析图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
    except ImportError:
        print("⚠ matplotlib not available, skipping plots")
        return
    
    if df.empty:
        print(f"No data to plot for {param_name}")
        return
    
    df_sorted = df.sort_values(param_name)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=14, fontweight='bold')
    
    # AUPRC
    axes[0, 0].plot(df_sorted[param_name], df_sorted['AUPRC'], 
                    marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0, 0].set_xlabel(param_name)
    axes[0, 0].set_ylabel('AUPRC')
    axes[0, 0].set_title('AUPRC vs ' + param_name)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([df_sorted['AUPRC'].min() * 0.95, df_sorted['AUPRC'].max() * 1.05])
    
    # F1 Score
    axes[0, 1].plot(df_sorted[param_name], df_sorted['F1'],
                    marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[0, 1].set_xlabel(param_name)
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score vs ' + param_name)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([df_sorted['F1'].min() * 0.95, df_sorted['F1'].max() * 1.05])
    
    # Precision
    axes[1, 0].plot(df_sorted[param_name], df_sorted['Precision'],
                     marker='^', linewidth=2, markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel(param_name)
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs ' + param_name)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([df_sorted['Precision'].min() * 0.95, df_sorted['Precision'].max() * 1.05])
    
    # Recall
    axes[1, 1].plot(df_sorted[param_name], df_sorted['Recall'],
                     marker='d', linewidth=2, markersize=8, color='#C73E1D')
    axes[1, 1].set_xlabel(param_name)
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall vs ' + param_name)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([df_sorted['Recall'].min() * 0.95, df_sorted['Recall'].max() * 1.05])
    
    plt.tight_layout()
    
    if save_plot:
        output_file = f'sensitivity_analysis_{param_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    """主函数"""
    print("="*80)
    print("Results Analysis Script")
    print("="*80)
    
    # 收集消融实验结果
    print("\n1. Collecting ablation experiment results...")
    ablation_df = collect_ablation_results()
    if not ablation_df.empty:
        generate_ablation_report(ablation_df)
    else:
        print("⚠ No ablation results found. Run ablation experiments first.")
    
    # 收集敏感性分析结果
    print("\n2. Collecting sensitivity analysis results...")
    sensitivity_df = collect_sensitivity_results('weight_loss_ms_disc')
    if not sensitivity_df.empty:
        generate_sensitivity_report(sensitivity_df, 'weight_loss_ms_disc')
        plot_sensitivity_analysis(sensitivity_df, 'weight_loss_ms_disc')
    else:
        print("⚠ No sensitivity analysis results found. Run sensitivity analysis first.")
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)


if __name__ == '__main__':
    main()

