"""
参数敏感性分析自动化脚本
==========================
功能：自动运行参数敏感性分析实验

使用方法：
python main_new/run_sensitivity_analysis.py
"""

import os
import sys
import subprocess
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_sensitivity_experiment(param_name, param_value, base_config, src='F-5', trg='C-1'):
    """
    运行单个敏感性分析实验
    
    参数:
        param_name: 参数名称（如 'weight_loss_ms_disc'）
        param_value: 参数值
        base_config: 基础配置字典
        src: 源域ID
        trg: 目标域ID
    """
    config = base_config.copy()
    config[param_name] = param_value
    
    exp_name = f"{param_name}_{param_value}"
    config['experiment_folder'] = f"MSL_Sensitivity_{param_name}_{param_value}"
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Parameter: {param_name} = {param_value}")
    print(f"{'='*60}\n")
    
    # 检查结果是否已存在
    result_dir = os.path.join('results', config['experiment_folder'], f'{src}-{trg}')
    if os.path.exists(result_dir):
        log_file = os.path.join(result_dir, 'eval_train.log')
        if os.path.exists(log_file):
            print(f"⚠ Results already exist, skipping...")
            return True
    
    # 构建训练命令
    train_cmd = [
        'python', 'main_new/train.py',
        '--algo_name', 'newmodel',
        '--num_epochs', '20',
        '--batch_size', '256',
        '--eval_batch_size', '256',
        '--learning_rate', '1e-4',
        '--dropout', '0.1',
        '--weight_decay', '1e-4',
        '--num_channels_TCN', '128-256-512',
        '--dilation_factor_TCN', '3',
        '--kernel_size_TCN', '7',
        '--hidden_dim_MLP', '1024',
        '--queue_size', '98304',
        '--momentum', '0.99',
        '--id_src', src,
        '--id_trg', trg,
        '--path_src', 'datasets/MSL_SMAP',
        '--path_trg', 'datasets/MSL_SMAP',
        '--experiment_folder', config['experiment_folder'],
        '--seed', '1234',
    ]
    
    # 添加所有配置参数
    for key, val in config.items():
        if key != 'experiment_folder':
            train_cmd.extend([f'--{key}', str(val)])
    
    # 运行训练
    print(f"Training {exp_name}...")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Training failed")
        print(result.stderr[:500])  # 只打印前500字符
        return False
    
    print(f"✓ Training completed")
    
    # 运行评估
    print(f"Evaluating {exp_name}...")
    eval_cmd = [
        'python', 'main_new/eval.py',
        '--experiments_main_folder', 'results',
        '--experiment_folder', config['experiment_folder'],
        '--id_src', src,
        '--id_trg', trg,
    ]
    
    result = subprocess.run(eval_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Evaluation failed")
        print(result.stderr[:500])
        return False
    
    print(f"✓ Experiment completed successfully\n")
    return True


def analyze_sensitivity_1_weight_loss_ms_disc():
    """敏感性分析1：多尺度域对抗损失权重"""
    print("="*80)
    print("Sensitivity Analysis 1: weight_loss_ms_disc")
    print("="*80)
    
    base_config = {
        'weight_loss_disc': 0.5,
        'weight_loss_pred': 1.0,
        'weight_loss_src_sup': 0.1,
        'weight_loss_trg_inj': 0.1,
    }
    
    # 参数值范围
    param_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = {}
    for value in param_values:
        success = run_sensitivity_experiment(
            'weight_loss_ms_disc',
            value,
            base_config
        )
        results[value] = 'Success' if success else 'Failed'
    
    return results


def analyze_sensitivity_2_tcn_layers():
    """敏感性分析2：TCN层数配置"""
    print("="*80)
    print("Sensitivity Analysis 2: TCN Layer Configurations")
    print("="*80)
    
    base_config = {
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'weight_loss_pred': 1.0,
        'weight_loss_src_sup': 0.1,
        'weight_loss_trg_inj': 0.1,
    }
    
    # 不同的TCN层配置
    layer_configs = {
        '128-256': '2层（低-中）',
        '256-512': '2层（中-高）',
        '128-256-512': '3层（默认）',
        '128-256-512-1024': '4层（扩展）',
    }
    
    results = {}
    for layer_config, description in layer_configs.items():
        config = base_config.copy()
        config['num_channels_TCN'] = layer_config
        config['experiment_folder'] = f"MSL_Sensitivity_TCN_{layer_config.replace('-', '_')}"
        
        print(f"\nTesting: {layer_config} ({description})")
        
        # 构建训练命令
        train_cmd = [
            'python', 'main_new/train.py',
            '--algo_name', 'newmodel',
            '--num_epochs', '20',
            '--batch_size', '256',
            '--learning_rate', '1e-4',
            '--num_channels_TCN', layer_config,
            '--weight_loss_ms_disc', '0.3',
            '--id_src', 'F-5',
            '--id_trg', 'C-1',
            '--experiment_folder', config['experiment_folder'],
        ]
        
        for key, val in base_config.items():
            if key != 'num_channels_TCN':
                train_cmd.extend([f'--{key}', str(val)])
        
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        success = result.returncode == 0
        
        if success:
            # 运行评估
            eval_cmd = [
                'python', 'main_new/eval.py',
                '--experiment_folder', config['experiment_folder'],
                '--id_src', 'F-5',
                '--id_trg', 'C-1',
            ]
            subprocess.run(eval_cmd)
        
        results[layer_config] = 'Success' if success else 'Failed'
    
    return results


def main():
    """主函数"""
    print("="*80)
    print("Multi-Scale Domain Adversarial DACAD - Sensitivity Analysis")
    print("="*80)
    
    all_results = {}
    start_time = datetime.now()
    
    # 分析1：多尺度损失权重敏感性
    print("\n" + "="*80)
    print("Starting Analysis 1: weight_loss_ms_disc")
    print("="*80)
    results_1 = analyze_sensitivity_1_weight_loss_ms_disc()
    all_results['weight_loss_ms_disc'] = results_1
    
    # 分析2：TCN层数配置（可选，需要较长时间）
    # print("\n" + "="*80)
    # print("Starting Analysis 2: TCN Layer Configurations")
    # print("="*80)
    # results_2 = analyze_sensitivity_2_tcn_layers()
    # all_results['tcn_layers'] = results_2
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    # 保存结果
    summary = {
        'timestamp': datetime.now().isoformat(),
        'duration_hours': duration,
        'results': all_results,
    }
    
    summary_file = 'sensitivity_analysis_results.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*80)
    print("Sensitivity Analysis Summary")
    print("="*80)
    print(f"Duration: {duration:.2f} hours")
    print(f"\nResults saved to: {summary_file}")
    print("\nDetailed Results:")
    print("-"*80)
    for analysis_name, results in all_results.items():
        print(f"\n{analysis_name}:")
        for param_value, status in results.items():
            status_icon = "✓" if status == 'Success' else "❌"
            print(f"  {status_icon} {param_value}: {status}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

