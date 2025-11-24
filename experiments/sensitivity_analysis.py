#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSPAD参数敏感性分析脚本
========================
功能：自动运行参数敏感性分析实验，找到最优参数配置

使用方法：
    # 运行所有敏感性分析
    python experiments/sensitivity_analysis.py
    
    # 运行特定参数的分析
    python experiments/sensitivity_analysis.py --param weight_loss_ms_disc
    python experiments/sensitivity_analysis.py --param prototypical_margin
    
    # 指定数据集和源-目标对
    python experiments/sensitivity_analysis.py --dataset MSL --src F-5 --trg C-1
    
    # 如果GPU显存不足，可以减小batch_size和queue_size
    python experiments/sensitivity_analysis.py --batch-size 64 --queue-size 24576
    

多进程运行：
    - 可以在同一GPU上同时运行多个脚本实例
    - torch.cuda.empty_cache() 只清理当前进程的缓存，不会影响其他进程
    - 每个进程独立管理自己的GPU缓存，互不干扰
"""

import os
import sys
import subprocess
import argparse
import json
import pandas as pd
import re
from datetime import datetime
from typing import Dict, Any, Optional
import torch
from sklearn.metrics import roc_curve, auc

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 颜色定义
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


def print_colored(message: str, color: str = Colors.NC):
    """打印彩色消息"""
    print(f"{color}{message}{Colors.NC}")


def calculate_auroc_from_predictions(pred_file):
    """从预测文件中计算AUROC"""
    try:
        if not os.path.exists(pred_file):
            print_colored(f"Warning: Prediction file not found: {pred_file}", Colors.YELLOW)
            return None

        df = pd.read_csv(pred_file)

        if 'y' not in df.columns or 'y_pred' not in df.columns:
            print_colored(f"Warning: Required columns not found in {pred_file}", Colors.YELLOW)
            return None

        y_true = df['y'].values
        y_scores = df['y_pred'].values

        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        return roc_auc

    except Exception as e:
        print_colored(f"Error calculating AUROC for {pred_file}: {e}", Colors.YELLOW)
        return None


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

            # 提取Best F1 Score
            f1_match = re.search(r'Best F1 score is\s*:\s*([\d.]+)', content)
            if f1_match:
                metrics['Best_F1'] = float(f1_match.group(1))

            # 提取Precision
            prec_match = re.search(r'Best Prec score is\s*:\s*([\d.]+)', content)
            if prec_match:
                metrics['Precision'] = float(prec_match.group(1))

            # 提取Recall
            rec_match = re.search(r'Best Rec score is\s*:\s*([\d.]+)', content)
            if rec_match:
                metrics['Recall'] = float(rec_match.group(1))

    except Exception as e:
        print_colored(f"Error reading {log_file}: {e}", Colors.YELLOW)

    return metrics


def clear_gpu_cache():
    """
    清理GPU显存缓存
    
    注意：
        - torch.cuda.empty_cache() 只清理当前进程的PyTorch缓存，不会影响其他进程
        - 每个进程只清理自己的缓存，不会干扰其他进程正在使用的GPU内存
        - 可以在同一GPU上安全地同时运行多个进程
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 参数敏感性分析配置
SENSITIVITY_CONFIGS = {
    'weight_loss_ms_disc': {
        'name': 'Multi-Scale Domain Adversarial Loss Weight',
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 5.0],
        'base_config': {
            'weight_loss_disc': 0.0,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
        },
    },
    'prototypical_margin': {
        'name': 'Prototypical Network Margin',
        'values': [0.01, 0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 5.0, 10.0, 50.0],
        'base_config': {
            'weight_loss_disc': 0.0,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'scale_weights': [0.1, 0.3, 0.6],
        },
    },
    'weight_loss_src_sup': {
        'name': 'Source Supervised Contrastive Loss Weight',
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 5.0],
        'base_config': {
            'weight_loss_disc': 0.0,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
        },
    },
    'weight_loss_trg_inj': {
        'name': 'Target Injection Contrastive Loss Weight',
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 1.5, 5.0],
        'base_config': {
            'weight_loss_disc': 0.0,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
        },
    },
    'scale_weights': {
        'name': 'Multi-Scale Layer Weights',
        'values': [
            [0.1, 0.3, 0.6],
            [0.33, 0.33, 0.34],
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.2, 0.4, 0.4],
            [0.5, 0.3, 0.2],
            [0.0, 0.5, 0.5],
        ],
        'base_config': {
            'weight_loss_disc': 0.0,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
        },
    },
}

# 数据集配置（减小batch_size以节省显存）
DATASET_CONFIGS = {
    "MSL": {
        "path_src": "datasets/MSL_SMAP",
        "path_trg": "datasets/MSL_SMAP",
        "batch_size": 256,  
        "dropout": 0.1,
        "num_channels_TCN": "128-256-512",
        "hidden_dim_MLP": 1024,
    },
    "SMD": {
        "path_src": "datasets/SMD/test",
        "path_trg": "datasets/SMD/test",
        "batch_size": 128,  
        "dropout": 0.1,
        "num_channels_TCN": "128-256-512",
        "hidden_dim_MLP": 1024,
    },
    "Boiler": {
        "path_src": "datasets/Boiler",
        "path_trg": "datasets/Boiler",
        "batch_size": 256,
        "dropout": 0.2,
        "num_channels_TCN": "128-128-128",
        "hidden_dim_MLP": 256,
    },
    "FWUAV": {
        "path_src": "datasets/FWUAV",
        "path_trg": "datasets/FWUAV",
        "batch_size": 128,
        "dropout": 0.1,
        "num_channels_TCN": "64-128-256",
        "hidden_dim_MLP": 512,
    },
    "ALFA": {
        "path_src": "datasets/ALFA",
        "path_trg": "datasets/ALFA",
        "batch_size": 128,
        "dropout": 0.1,
        "num_channels_TCN": "64-128-256",
        "hidden_dim_MLP": 512,
    },
    "UAV": {
        "path_src": "datasets/UAV",
        "path_trg": "datasets/UAV",
        "batch_size": 128,
        "dropout": 0.1,
        "num_channels_TCN": "64-128-256",
        "hidden_dim_MLP": 512,
    },
}

# 默认训练参数
DEFAULT_TRAIN_PARAMS = {
    "learning_rate": "1e-4",
    "dilation_factor_TCN": "3",
    "kernel_size_TCN": "7",
    "queue_size": "98304",
    "momentum": "0.99",
}

# 路径常量
RESULTS_DIR = "results"
SUMMARY_DIR = "experiment_results"
SENSITIVITY_DIR = os.path.join(SUMMARY_DIR, "参数敏感性分析实验")


def format_param_value(param_value: Any) -> str:
    """格式化参数值为字符串（用于文件夹名）"""
    if isinstance(param_value, list):
        return '_'.join([str(v) for v in param_value])
    return str(param_value).replace('.', '_')


def get_summary_csv_path(dataset: str, src: str) -> str:
    """获取汇总CSV文件路径"""
    dataset_lower = dataset.lower()
    fname_map = {
        "msl": f'MSPAD_MSL_{src}.csv',
        "smd": f'MSPAD_SMD_{src}.csv',
        "boiler": f'MSPAD_Boiler_{src}.csv',
        "fwuav": f'MSPAD_FWUAV_{src}.csv',
        "uav": f'MSPAD_UAV_{src}.csv',
    }
    fname = fname_map.get(dataset_lower, f'MSPAD_test_{src}.csv')
    return os.path.join(SUMMARY_DIR, fname)


def is_experiment_completed(exp_folder: str, src: str, trg: str) -> bool:
    """检查实验是否已完成"""
    result_dir = os.path.join(RESULTS_DIR, exp_folder, f"{src}-{trg}")
    model_file = os.path.join(result_dir, "model_best.pth.tar")
    pred_file = os.path.join(result_dir, "predictions_test_target.csv")
    return os.path.exists(model_file) and os.path.exists(pred_file)


def build_train_command(
    param_name: str,
    param_value: Any,
    base_config: dict,
    dataset: str,
    dataset_config: dict,
    src: str,
    trg: str,
    num_epochs: int,
    seed: int,
) -> tuple:
    """构建训练命令"""
    exp_value_str = format_param_value(param_value)
    exp_folder = f"{dataset}_Sensitivity_{param_name}_{exp_value_str}"
    
    # 获取eval_batch_size，如果未设置则使用batch_size
    eval_batch_size = dataset_config.get("eval_batch_size", dataset_config["batch_size"])
    
    cmd = [
        "python", "main_new/train.py",
        "--algo_name", "MSPAD",
        "--num_epochs", str(num_epochs),
        "--batch_size", str(dataset_config["batch_size"]),
        "--eval_batch_size", str(eval_batch_size),
        "--learning_rate", DEFAULT_TRAIN_PARAMS["learning_rate"],
        "--dropout", str(dataset_config["dropout"]),
        "--num_channels_TCN", dataset_config["num_channels_TCN"],
        "--dilation_factor_TCN", DEFAULT_TRAIN_PARAMS["dilation_factor_TCN"],
        "--kernel_size_TCN", DEFAULT_TRAIN_PARAMS["kernel_size_TCN"],
        "--hidden_dim_MLP", str(dataset_config["hidden_dim_MLP"]),
        "--queue_size", DEFAULT_TRAIN_PARAMS["queue_size"],
        "--momentum", DEFAULT_TRAIN_PARAMS["momentum"],
        "--id_src", src,
        "--id_trg", trg,
        "--path_src", dataset_config["path_src"],
        "--path_trg", dataset_config["path_trg"],
        "--experiments_main_folder", RESULTS_DIR,
        "--experiment_folder", exp_folder,
        "--seed", str(seed),
    ]
    
    # 添加配置参数（跳过scale_weights和当前分析的参数）
    config = base_config.copy()
    for key, val in config.items():
        if key == 'scale_weights':
            if param_name == 'scale_weights':
                print_colored("⚠ Warning: scale_weights parameter is not yet supported via command line", Colors.YELLOW)
                print_colored("   Model will use default scale_weights: [0.1, 0.3, 0.6]", Colors.YELLOW)
            continue
        if key != param_name:
            cmd.extend([f"--{key}", str(val)])
    
    # 添加当前分析的参数值
    if param_name != 'scale_weights':
        cmd.extend([f"--{param_name}", str(param_value)])
    
    return cmd, exp_folder


def run_command(cmd: list, task_name: str) -> bool:
    """运行命令并处理错误"""
    print_colored(f"{task_name}...", Colors.BLUE)
    # 运行前清理GPU缓存
    clear_gpu_cache()
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        # 运行后清理GPU缓存
        clear_gpu_cache()
        if result.returncode != 0:
            print_colored(f"❌ {task_name} failed", Colors.RED)
            # 显示错误信息（优先显示stderr，如果没有则显示stdout）
            error_msg = result.stderr if result.stderr else result.stdout
            if error_msg:
                # 显示完整的错误信息，但限制长度
                error_lines = error_msg.split('\n')
                # 显示最后20行错误信息
                display_lines = error_lines[-20:] if len(error_lines) > 20 else error_lines
                print_colored('\n'.join(display_lines), Colors.RED)
            return False
        print_colored(f"✓ {task_name} completed", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"❌ {task_name} error: {e}", Colors.RED)
        clear_gpu_cache()
        return False


def find_experiment_results_dir(exp_folder: str, preferred_src_trg: str = None) -> str:
    """查找实验结果目录
    
    Args:
        exp_folder: 实验文件夹名
        preferred_src_trg: 首选的源-目标对（如"002-026"）
    
    Returns:
        实际包含结果的目录名，如果没找到返回None
    """
    exp_path = os.path.join(RESULTS_DIR, exp_folder)
    if not os.path.exists(exp_path):
        return None
    
    # 如果首选目录存在，直接返回
    if preferred_src_trg and os.path.exists(os.path.join(exp_path, preferred_src_trg)):
        return preferred_src_trg
    
    # 扫描所有子目录，找到包含eval_train.log的目录
    for item in os.listdir(exp_path):
        item_path = os.path.join(exp_path, item)
        if os.path.isdir(item_path):
            log_file = os.path.join(item_path, "eval_train.log")
            pred_file = os.path.join(item_path, "predictions_test_target.csv")
            if os.path.exists(log_file) and os.path.exists(pred_file):
                return item
    
    return None


def save_sensitivity_result(
    dataset: str,
    src: str,
    trg: str,
    param_name: str,
    param_value: Any,
    exp_folder: str,
) -> bool:
    """保存敏感性分析结果到CSV文件"""
    try:
        # 查找实际的实验结果目录
        actual_src_trg = find_experiment_results_dir(exp_folder, f"{src}-{trg}")
        if not actual_src_trg:
            print_colored(f"⚠ Warning: No experiment results found in {exp_folder}", Colors.YELLOW)
            return False

        # 构建文件路径
        pred_file = os.path.join(RESULTS_DIR, exp_folder, actual_src_trg, "predictions_test_target.csv")
        log_file = os.path.join(RESULTS_DIR, exp_folder, actual_src_trg, "eval_train.log")

        print_colored(f"Using results from: {actual_src_trg}", Colors.BLUE)

        # 计算AUROC
        auroc = calculate_auroc_from_predictions(pred_file)

        # 提取其他指标
        metrics = extract_metrics_from_log(log_file)
        auprc = metrics.get('AUPRC')
        best_f1 = metrics.get('Best_F1')

        # 准备结果数据
        result_data = {
            'dataset': dataset,
            'src_trg': actual_src_trg,  # 使用实际的源-目标对
            'param_name': param_name,
            'param_value': format_param_value(param_value),
            'exp_folder': exp_folder,
            'AUROC': auroc if auroc is not None else float('nan'),
            'AUPRC': auprc if auprc is not None else float('nan'),
            'Best_F1': best_f1 if best_f1 is not None else float('nan'),
        }

        # 保存到参数敏感性分析实验文件夹
        os.makedirs(SENSITIVITY_DIR, exist_ok=True)
        sensitivity_csv = os.path.join(SENSITIVITY_DIR, f'Sensitivity_{dataset}_{src}_{trg}.csv')

        # 追加或创建文件
        mode = 'a' if os.path.exists(sensitivity_csv) else 'w'
        header = mode == 'w'

        df = pd.DataFrame([result_data])
        df.to_csv(sensitivity_csv, mode=mode, header=header, index=False)

        print_colored(f"✓ 结果已保存到: {sensitivity_csv}", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"⚠ Warning: Failed to save sensitivity results: {e}", Colors.YELLOW)
        return False


def run_sensitivity_experiment(
    param_name: str,
    param_value: Any,
    base_config: dict,
    dataset: str,
    src: str,
    trg: str,
    num_epochs: int,
    seed: int,
    skip_if_completed: bool = True,
) -> bool:
    """运行单个敏感性分析实验"""
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"Parameter: {param_name} = {param_value}", Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)
    
    # 实验开始前清理GPU缓存
    clear_gpu_cache()
    
    # 检查数据集配置
    dataset_config = DATASET_CONFIGS.get(dataset)
    if not dataset_config:
        print_colored(f"Error: Unknown dataset: {dataset}", Colors.RED)
        return False
    
    # 构建训练命令
    train_cmd, exp_folder = build_train_command(
        param_name, param_value, base_config, dataset,
        dataset_config, src, trg, num_epochs, seed
    )
    
    # 检查是否已完成
    if skip_if_completed and is_experiment_completed(exp_folder, src, trg):
        print_colored("⏭  Skipped (already completed)", Colors.YELLOW)
        clear_gpu_cache()
        return True
    
    # 运行训练和评估
    if not run_command(train_cmd, "Training"):
        clear_gpu_cache()
        return False
    
    # 训练和评估之间清理GPU缓存
    clear_gpu_cache()
    
    eval_cmd = [
        "python", "main_new/eval.py",
        "--experiments_main_folder", RESULTS_DIR,
        "--experiment_folder", exp_folder,
        "--id_src", src,
        "--id_trg", trg,
    ]
    
    if not run_command(eval_cmd, "Evaluating"):
        clear_gpu_cache()
        return False
    
    # 保存结果
    save_sensitivity_result(dataset, src, trg, param_name, param_value, exp_folder)
    
    # 实验结束后清理GPU缓存
    clear_gpu_cache()
    
    print_colored("✓ Experiment completed successfully", Colors.GREEN)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="MSPAD参数敏感性分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dataset", type=str, default="MSL",
                       choices=["MSL", "SMD", "Boiler", "FWUAV", "ALFA", "UAV"],
                       help="Dataset name (default: MSL)")
    parser.add_argument("--src", type=str, default="F-5",
                       help="Source domain ID (default: F-5)")
    parser.add_argument("--trg", type=str, default="C-1",
                       help="Target domain ID (default: C-1)")
    parser.add_argument("--param", type=str,
                       choices=list(SENSITIVITY_CONFIGS.keys()) + ['all'],
                       default="all",
                       help="Parameter to analyze (default: all)")
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="Number of epochs (default: 20)")
    parser.add_argument("--seed", type=int, default=2021,
                       help="Random seed (default: 2021)")
    parser.add_argument("--skip-completed", action="store_true", default=False,
                       help="Skip completed experiments (default: False)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch_size (default: use dataset default)")
    parser.add_argument("--queue-size", type=int, default=None,
                       help="Override queue_size (default: 49152)")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                       help="Override eval_batch_size (default: same as batch_size)")
    
    args = parser.parse_args()
    
    # 应用用户指定的batch_size和queue_size覆盖
    if args.batch_size is not None:
        for dataset_config in DATASET_CONFIGS.values():
            dataset_config["batch_size"] = args.batch_size
        print_colored(f"Using custom batch_size: {args.batch_size}", Colors.YELLOW)
    
    if args.queue_size is not None:
        DEFAULT_TRAIN_PARAMS["queue_size"] = str(args.queue_size)
        print_colored(f"Using custom queue_size: {args.queue_size}", Colors.YELLOW)
    
    if args.eval_batch_size is not None:
        for dataset_config in DATASET_CONFIGS.values():
            dataset_config["eval_batch_size"] = args.eval_batch_size
        print_colored(f"Using custom eval_batch_size: {args.eval_batch_size}", Colors.YELLOW)
    
    # 打印启动信息
    print_colored("="*80, Colors.CYAN)
    print_colored("MSPAD参数敏感性分析", Colors.CYAN)
    print_colored("="*80, Colors.CYAN)
    print_colored(f"Dataset: {args.dataset}", Colors.CYAN)
    print_colored(f"Source: {args.src} -> Target: {args.trg}", Colors.CYAN)
    print_colored(f"Parameter: {args.param}", Colors.CYAN)
    print_colored(f"Epochs: {args.num_epochs}", Colors.CYAN)
    print_colored(f"Seed: {args.seed}", Colors.CYAN)
    print_colored(f"Batch Size: {DATASET_CONFIGS[args.dataset]['batch_size']}", Colors.CYAN)
    print_colored(f"Queue Size: {DEFAULT_TRAIN_PARAMS['queue_size']}", Colors.CYAN)
    if torch.cuda.is_available():
        print_colored(f"GPU: {torch.cuda.get_device_name(0)}", Colors.CYAN)
        print_colored(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", Colors.CYAN)
    print_colored("="*80 + "\n", Colors.CYAN)
    
    # 选择要分析的参数
    params_to_analyze = (
        list(SENSITIVITY_CONFIGS.keys())
        if args.param == "all"
        else [args.param]
    )
    
    # 运行敏感性分析
    all_results = {}
    start_time = datetime.now()
    
    for param_name in params_to_analyze:
        if param_name not in SENSITIVITY_CONFIGS:
            print_colored(f"Warning: Unknown parameter '{param_name}', skipping...", Colors.YELLOW)
            continue
        
        param_config = SENSITIVITY_CONFIGS[param_name]
        print_colored(f"\n{'='*80}", Colors.MAGENTA)
        print_colored(f"Analyzing: {param_config['name']}", Colors.MAGENTA)
        print_colored(f"{'='*80}", Colors.MAGENTA)
        
        results = {}
        for param_value in param_config['values']:
            try:
                success = run_sensitivity_experiment(
                    param_name=param_name,
                    param_value=param_value,
                    base_config=param_config['base_config'],
                    dataset=args.dataset,
                    src=args.src,
                    trg=args.trg,
                    num_epochs=args.num_epochs,
                    seed=args.seed,
                    skip_if_completed=args.skip_completed,
                )
                results[str(param_value)] = 'Success' if success else 'Failed'
                # 每个参数值实验后清理GPU缓存
                clear_gpu_cache()
            except Exception as e:
                print_colored(f"❌ Error: {str(e)}", Colors.RED)
                results[str(param_value)] = 'Error'
                clear_gpu_cache()
        
        all_results[param_name] = {
            'name': param_config['name'],
            'results': results,
        }
    
    # 保存结果摘要
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'duration_hours': duration,
        'dataset': args.dataset,
        'source': args.src,
        'target': args.trg,
        'results': all_results,
    }
    
    # 确保结果文件夹存在
    os.makedirs(SENSITIVITY_DIR, exist_ok=True)
    
    # 保存JSON摘要文件到参数敏感性分析实验文件夹
    summary_file = os.path.join(
        SENSITIVITY_DIR,
        f'sensitivity_analysis_{args.dataset}_{args.src}_{args.trg}.json'
    )
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print_colored("\n" + "="*80, Colors.GREEN)
    print_colored("Sensitivity Analysis Summary", Colors.GREEN)
    print_colored("="*80, Colors.GREEN)
    print_colored(f"Duration: {duration:.2f} hours", Colors.GREEN)
    print_colored(f"\n所有结果文件已保存到: {SENSITIVITY_DIR}", Colors.GREEN)
    print_colored(f"  - JSON摘要文件: {os.path.basename(summary_file)}", Colors.GREEN)
    
    # 列出所有CSV结果文件
    csv_files = [f for f in os.listdir(SENSITIVITY_DIR) if f.endswith('.csv')]
    if csv_files:
        print_colored(f"  - CSV结果文件 ({len(csv_files)}个):", Colors.GREEN)
        for csv_file in sorted(csv_files):
            print_colored(f"    * {csv_file}", Colors.GREEN)
    print_colored("\nDetailed Results:", Colors.GREEN)
    print_colored("-"*80, Colors.GREEN)
    for param_name, param_results in all_results.items():
        print_colored(f"\n{param_results['name']}:", Colors.GREEN)
        for param_value, status in param_results['results'].items():
            status_icon = "✓" if status == 'Success' else "❌"
            print_colored(f"  {status_icon} {param_value}: {status}", Colors.GREEN)
    print_colored("="*80 + "\n", Colors.GREEN)


if __name__ == '__main__':
    main()
