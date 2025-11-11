"""
消融实验自动化运行脚本
======================
功能：自动运行所有消融实验并收集结果

使用方法：
python main_new/run_ablation_experiments.py
"""

import os
import sys
import subprocess
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 实验配置
EXPERIMENTS = {
    'Baseline': {
        'algo_name': 'dacad',  # 使用原始DACAD
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.0,  # 无多尺度
        'experiment_folder': 'MSL_Baseline',
        'description': '原始DACAD（单尺度域对抗）'
    },
    'Exp-1.1': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'experiment_folder': 'MSL_MSDA_Full',
        'description': '多尺度DACAD（完整版：3层多尺度 + 单尺度）'
    },
    'Exp-1.2': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.0,  # 无单尺度
        'weight_loss_ms_disc': 0.3,
        'experiment_folder': 'MSL_MSDA_MSOnly',
        'description': '多尺度DACAD（仅多尺度，无单尺度）'
    },
    'Exp-2.1': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'use_layer_mask': [1, 0, 0],  # 仅第1层
        'experiment_folder': 'MSL_MSDA_Layer1Only',
        'description': '仅低层域对抗（Layer 1）'
    },
    'Exp-2.2': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'use_layer_mask': [0, 1, 0],  # 仅第2层
        'experiment_folder': 'MSL_MSDA_Layer2Only',
        'description': '仅中层域对抗（Layer 2）'
    },
    'Exp-2.3': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'use_layer_mask': [0, 0, 1],  # 仅第3层
        'experiment_folder': 'MSL_MSDA_Layer3Only',
        'description': '仅高层域对抗（Layer 3）'
    },
    'Exp-2.7': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'use_layer_mask': [1, 1, 1],  # 所有层
        'experiment_folder': 'MSL_MSDA_AllLayers',
        'description': '所有层域对抗（完整配置）'
    },
    'Exp-3.1': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'scale_weights': [0.1, 0.3, 0.6],  # 默认配置
        'experiment_folder': 'MSL_MSDA_Weights_Default',
        'description': '默认权重配置 [0.1, 0.3, 0.6]'
    },
    'Exp-3.2': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'scale_weights': [0.33, 0.33, 0.34],  # 均匀权重
        'experiment_folder': 'MSL_MSDA_Weights_Uniform',
        'description': '均匀权重配置 [0.33, 0.33, 0.34]'
    },
    'Exp-3.3': {
        'algo_name': 'newmodel',
        'weight_loss_disc': 0.5,
        'weight_loss_ms_disc': 0.3,
        'scale_weights': [0.6, 0.3, 0.1],  # 反向权重
        'experiment_folder': 'MSL_MSDA_Weights_Reverse',
        'description': '反向权重配置 [0.6, 0.3, 0.1]'
    },
}


def run_experiment(exp_name, config, src='F-5', trg='C-1', skip_if_exists=True):
    """
    运行单个实验
    
    参数:
        exp_name: 实验名称
        config: 实验配置字典
        src: 源域ID
        trg: 目标域ID
        skip_if_exists: 如果结果已存在是否跳过
    """
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Configuration: {config}")
    print(f"{'='*80}\n")
    
    # 检查结果是否已存在
    result_dir = os.path.join('results', config['experiment_folder'], f'{src}-{trg}')
    if skip_if_exists and os.path.exists(result_dir):
        log_file = os.path.join(result_dir, 'eval_train.log')
        if os.path.exists(log_file):
            print(f"⚠ Results already exist for {exp_name}, skipping...")
            return True
    
    # 构建训练命令
    train_cmd = [
        'python', 'main_new/train.py',
        '--algo_name', config['algo_name'],
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
        '--weight_loss_pred', '1.0',
        '--weight_loss_src_sup', '0.1',
        '--weight_loss_trg_inj', '0.1',
        '--weight_loss_disc', str(config['weight_loss_disc']),
        '--weight_loss_ms_disc', str(config['weight_loss_ms_disc']),
        '--id_src', src,
        '--id_trg', trg,
        '--path_src', 'datasets/MSL_SMAP',
        '--path_trg', 'datasets/MSL_SMAP',
        '--experiment_folder', config['experiment_folder'],
        '--seed', '1234',
    ]
    
    # 运行训练
    print(f"Training {exp_name}...")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Training failed for {exp_name}")
        print(result.stderr)
        return False
    
    print(f"✓ Training completed for {exp_name}")
    
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
        print(f"❌ Evaluation failed for {exp_name}")
        print(result.stderr)
        return False
    
    print(f"✓ Experiment {exp_name} completed successfully\n")
    return True


def main():
    """主函数：运行所有消融实验"""
    print("="*80)
    print("Multi-Scale Domain Adversarial DACAD - Ablation Experiments")
    print("="*80)
    
    # 实验参数
    src = 'F-5'
    trg = 'C-1'
    
    # 运行所有实验
    results = {}
    start_time = datetime.now()
    
    for exp_name, config in EXPERIMENTS.items():
        try:
            success = run_experiment(exp_name, config, src, trg)
            results[exp_name] = {
                'status': 'Success' if success else 'Failed',
                'description': config.get('description', 'N/A'),
                'config': {k: v for k, v in config.items() if k != 'description'}
            }
        except Exception as e:
            print(f"❌ Error in {exp_name}: {str(e)}")
            results[exp_name] = {
                'status': 'Error',
                'error': str(e)
            }
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600  # 小时
    
    # 保存结果摘要
    summary = {
        'timestamp': datetime.now().isoformat(),
        'duration_hours': duration,
        'source': src,
        'target': trg,
        'results': results,
        'summary': {
            'total': len(results),
            'success': sum(1 for r in results.values() if r['status'] == 'Success'),
            'failed': sum(1 for r in results.values() if r['status'] == 'Failed'),
            'errors': sum(1 for r in results.values() if r['status'] == 'Error'),
        }
    }
    
    summary_file = 'ablation_results_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    print(f"Total experiments: {summary['summary']['total']}")
    print(f"Successful: {summary['summary']['success']}")
    print(f"Failed: {summary['summary']['failed']}")
    print(f"Errors: {summary['summary']['errors']}")
    print(f"Duration: {duration:.2f} hours")
    print(f"\nResults saved to: {summary_file}")
    print("\nDetailed Results:")
    print("-"*80)
    for exp_name, result in results.items():
        status_icon = "✓" if result['status'] == 'Success' else "❌"
        print(f"{status_icon} {exp_name}: {result['status']}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()

