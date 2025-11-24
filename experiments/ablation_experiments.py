#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSPAD消融实验脚本
================
功能：自动运行所有消融实验，验证每个核心组件的贡献

实验设计：
核心组件消融（core）- 完整模型 + 消去各个组件后的模型：
   - Abl-Full: MSPAD Full（完整模型，所有组件）
   - Abl-w/o-MS-DA: 移除多尺度域对抗损失
   - Abl-w/o-Prototypical: 移除原型网络（使用Deep SVDD）
   - Abl-w/o-SrcSupCL: 移除源域监督对比损失
   - Abl-w/o-TrgInjCL: 移除目标域注入对比损失

注意：multi_scale 和 loss 组已移除，如需运行请取消代码中的注释

使用方法：
    # 运行核心组件消融实验（默认）
    python experiments/ablation_experiments.py --dataset ALFA --src 001 --all-targets
    python experiments/ablation_experiments.py --dataset FWUAV --src 1 --all-targets

    # 指定数据集和源-目标对
    python experiments/ablation_experiments.py --dataset MSL --src F-5 --trg C-1
    python experiments/ablation_experiments.py --dataset FWUAV --src 1 --trg 6

    # 只指定源域，对所有其他文件作为目标域运行（推荐）
    python experiments/ablation_experiments.py --dataset MSL --src F-5 --all-targets
    python experiments/ablation_experiments.py --dataset FWUAV --src 1 --all-targets

    # 跳过已完成的实验（断点续传）
    python experiments/ablation_experiments.py --dataset ALFA --src 001 --all-targets --skip-completed
"""

import os
import sys
import subprocess
import argparse
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, auc

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 路径常量
SUMMARY_DIR = "experiment_results"
ABLATION_DIR = os.path.join(SUMMARY_DIR, "ablation")

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


def find_experiment_results_dir(exp_folder: str, preferred_src_trg: str = None) -> str:
    """查找实验结果目录

    Args:
        exp_folder: 实验文件夹名
        preferred_src_trg: 首选的源-目标对（如"002-026"）

    Returns:
        实际包含结果的目录名，如果没找到返回None
    """
    exp_path = os.path.join("results", exp_folder)
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


# 消融实验配置
ABLATION_EXPERIMENTS = {
    # ========== 核心组件消融 ==========
    # 实验设计：完整模型 + 消去各个组件后的模型
    # 目标：验证每个核心组件的贡献
    'core': {
        'Abl-Full': {
            'name': 'MSPAD Full',
            'description': 'MSPAD完整版（所有组件）',
            'algo_name': 'MSPAD',
            # 注意：MSPAD不使用weight_loss_disc（单尺度域对抗损失），已由weight_loss_ms_disc替代
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_MSPAD_Full',
        },
        'Abl-w/o-MS-DA': {
            'name': 'w/o Multi-Scale DA',
            'description': '移除多尺度域对抗损失',
            'algo_name': 'MSPAD',
            'weight_loss_ms_disc': 0.0,  # 移除多尺度域对抗损失
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'experiment_folder': 'Ablation_NoMultiScaleDA',
        },
        'Abl-w/o-Prototypical': {
            'name': 'w/o Prototypical',
            'description': '移除原型网络分类损失（将weight_loss_pred设为0）',
            'algo_name': 'MSPAD',
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 0.0,  # 移除原型网络分类损失
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,  # 虽然不使用，但保留参数避免错误
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_NoPrototypical',
        },
        'Abl-w/o-SrcSupCL': {
            'name': 'w/o Source Sup CL',
            'description': '移除源域监督对比损失',
            'algo_name': 'MSPAD',
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.0,  # 移除源域监督对比损失
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_NoSourceSupCL',
        },
        'Abl-w/o-TrgInjCL': {
            'name': 'w/o Target Inj CL',
            'description': '移除目标域注入对比损失',
            'algo_name': 'MSPAD',
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.0,  # 移除目标域注入对比损失
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_NoTargetInjCL',
        },
    },
    
    # ========== 以下实验组已移除，仅保留核心组件消融 ==========
    # multi_scale 和 loss 组已完全移除，如需运行请从历史版本恢复
}


def get_msl_files() -> List[str]:
    """获取MSL数据集的所有文件列表"""
    csv_path = 'datasets/MSL_SMAP/labeled_anomalies.csv'
    if not os.path.exists(csv_path):
        print_colored(f"Error: MSL dataset CSV file not found: {csv_path}", Colors.RED)
        return []
    
    csv_reader = pd.read_csv(csv_path, delimiter=',')
    data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
    space_files = np.asarray(data_info['chan_id'])
    
    test_dir = 'datasets/MSL_SMAP/test'
    if not os.path.exists(test_dir):
        print_colored(f"Error: MSL test directory not found: {test_dir}", Colors.RED)
        return []
    
    all_files = os.listdir(test_dir)
    all_names = [name[:-4] for name in all_files if name.endswith('.npy')]
    
    files = [file for file in all_names if file in space_files]
    return sorted(files)


def get_smd_files() -> List[str]:
    """获取SMD数据集的所有文件列表"""
    test_dir = 'datasets/SMD/test'
    if not os.path.exists(test_dir):
        print_colored(f"Error: SMD dataset directory not found: {test_dir}", Colors.RED)
        return []
    
    files = []
    for file in os.listdir(test_dir):
        if file.startswith('machine-') and file.endswith('.txt'):
            file_id = file.replace('machine-', '').replace('.txt', '')
            files.append(file_id)
    
    return sorted(files)


def get_boiler_files() -> List[str]:
    """获取Boiler数据集的所有文件列表"""
    boiler_dir = 'datasets/Boiler'
    if not os.path.exists(boiler_dir):
        print_colored(f"Error: Boiler dataset directory not found: {boiler_dir}", Colors.RED)
        return []

    files = []
    for file in os.listdir(boiler_dir):
        if file.endswith('.csv'):
            file_id = file.replace('.csv', '')
            files.append(file_id)

    return sorted(files)


def get_fwuav_files() -> List[str]:
    """获取FWUAV数据集的所有文件列表"""
    fwuav_dir = 'datasets/FWUAV'
    if not os.path.exists(fwuav_dir):
        print_colored(f"Error: FWUAV dataset directory not found: {fwuav_dir}", Colors.RED)
        return []

    files = []
    for item in os.listdir(fwuav_dir):
        item_path = os.path.join(fwuav_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            files.append(item)

    return sorted(files)


def get_uav_files() -> List[str]:
    """获取UAV数据集的所有文件列表"""
    uav_dir = 'datasets/UAV'
    if not os.path.exists(uav_dir):
        print_colored(f"Error: UAV dataset directory not found: {uav_dir}", Colors.RED)
        return []

    files = []
    for item in os.listdir(uav_dir):
        item_path = os.path.join(uav_dir, item)
        if os.path.isdir(item_path) and item.startswith('flight_'):
            # 提取flight编号，如 'flight_002' -> '002'
            flight_id = item.replace('flight_', '')
            files.append(flight_id)

    return sorted(files)


def get_dataset_files(dataset: str) -> List[str]:
    """根据数据集名称获取文件列表"""
    if dataset == "MSL":
        return get_msl_files()
    elif dataset == "SMD":
        return get_smd_files()
    elif dataset == "Boiler":
        return get_boiler_files()
    elif dataset == "FWUAV":
        return get_fwuav_files()
    elif dataset == "UAV":
        return get_uav_files()
    else:
        print_colored(f"Unknown dataset: {dataset}", Colors.RED)
        return []


def get_dataset_config(dataset: str) -> dict:
    """获取数据集的配置参数"""
    configs = {
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
    return configs.get(dataset, {})


def save_ablation_result(
    dataset: str,
    src: str,
    trg: str,
    exp_id: str,
    exp_folder: str,
    group: str = "all",
) -> bool:
    """保存消融实验结果到CSV文件"""
    try:
        # 查找实际的实验结果目录
        actual_src_trg = find_experiment_results_dir(exp_folder, f"{src}-{trg}")
        if not actual_src_trg:
            print_colored(f"⚠ Warning: No experiment results found in {exp_folder}", Colors.YELLOW)
            return False

        # 构建预测文件路径
        pred_file = os.path.join("results", exp_folder, actual_src_trg, "predictions_test_target.csv")
        log_file = os.path.join("results", exp_folder, actual_src_trg, "eval_train.log")

        print_colored(f"Using results from: {actual_src_trg}", Colors.BLUE)

        # 计算AUROC
        auroc = calculate_auroc_from_predictions(pred_file)

        # 提取其他指标
        metrics = extract_metrics_from_log(log_file)
        auprc = metrics.get('AUPRC')
        best_f1 = metrics.get('Best_F1')

        # 解析实际的源-目标对
        if '-' in actual_src_trg:
            actual_src, actual_trg = actual_src_trg.split('-', 1)
        else:
            actual_src, actual_trg = src, trg

        # 准备结果数据（每个src-trg对作为一行）
        result_data = {
            'src_id': actual_src,
            'trg_id': actual_trg,
            'exp_id': exp_id,
            'exp_folder': exp_folder,
            'AUROC': auroc if auroc is not None else float('nan'),
            'AUPRC': auprc if auprc is not None else float('nan'),
            'Best_F1': best_f1 if best_f1 is not None else float('nan'),
        }

        # 保存到消融实验文件夹（统一命名：Ablation_{group}_{dataset}_{src}.csv）
        os.makedirs(ABLATION_DIR, exist_ok=True)
        ablation_csv = os.path.join(ABLATION_DIR, f'Ablation_{group}_{dataset}_{src}.csv')

        # 追加或创建文件
        mode = 'a' if os.path.exists(ablation_csv) else 'w'
        header = mode == 'w'

        df = pd.DataFrame([result_data])
        df.to_csv(ablation_csv, mode=mode, header=header, index=False)

        print_colored(f"✓ 结果已保存到: {ablation_csv}", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"⚠ Warning: Failed to save ablation results: {e}", Colors.YELLOW)
        return False


def run_ablation_experiment(
    exp_id: str,
    config: dict,
    dataset: str = "MSL",
    src: str = "F-5",
    trg: str = "C-1",
    num_epochs: int = 20,
    seed: int = 1234,
    skip_if_completed: bool = True,
    group: str = "all",
) -> bool:
    """运行单个消融实验"""
    
    print_colored(f"\n{'='*80}", Colors.CYAN)
    print_colored(f"Experiment: {exp_id} - {config['name']}", Colors.CYAN)
    print_colored(f"Description: {config['description']}", Colors.CYAN)
    print_colored(f"{'='*80}", Colors.CYAN)
    
    dataset_config = get_dataset_config(dataset)
    if not dataset_config:
        print_colored(f"Error: Unknown dataset: {dataset}", Colors.RED)
        return False
    
    # 使用ablation/前缀，统一管理消融实验的详细结果
    exp_folder = f"ablation/{dataset}_{config['experiment_folder']}"
    result_dir = os.path.join("results", exp_folder, f"{src}-{trg}")
    
    # 检查是否已完成
    if skip_if_completed:
        model_file = os.path.join(result_dir, "model_best.pth.tar")
        pred_file = os.path.join(result_dir, "predictions_test_target.csv")
        if os.path.exists(model_file) and os.path.exists(pred_file):
            print_colored(f"⏭  Skipped (already completed): {exp_id}", Colors.YELLOW)
            return True
    
    algo_name = config['algo_name']
    
    # 确定训练和评估脚本
    if algo_name == 'dacad':
        train_script = "main/train.py"
        eval_script = "main/eval.py"
    elif algo_name == 'MSPAD':
        train_script = "main_new/train.py"
        eval_script = "main_new/eval.py"
    else:
        print_colored(f"Error: Unknown algorithm: {algo_name}", Colors.RED)
        return False
    
    # 构建训练命令
    train_cmd = [
        "python", train_script,
        "--algo_name", algo_name,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(dataset_config["batch_size"]),
        "--eval_batch_size", str(dataset_config["batch_size"]),
        "--learning_rate", "1e-4",
        "--dropout", str(dataset_config["dropout"]),
        "--num_channels_TCN", dataset_config["num_channels_TCN"],
        "--dilation_factor_TCN", "3",
        "--kernel_size_TCN", "7",
        "--stride_TCN", "1",  # 添加缺失的参数
        "--hidden_dim_MLP", str(dataset_config["hidden_dim_MLP"]),
        "--queue_size", "98304",
        "--momentum", "0.99",
        "--weight_decay", "1e-4",  # 添加缺失的参数
        "--weight_loss_pred", str(config['weight_loss_pred']),
        "--weight_loss_src_sup", str(config['weight_loss_src_sup']),
        "--weight_loss_trg_inj", str(config['weight_loss_trg_inj']),
        "--id_src", src,
        "--id_trg", trg,
        "--path_src", dataset_config["path_src"],
        "--path_trg", dataset_config["path_trg"],
        "--experiments_main_folder", "results",
        "--experiment_folder", exp_folder,
        "--seed", str(seed),
    ]
    
    # 添加算法特定参数
    if algo_name == 'dacad':
        # DACAD使用单尺度域对抗损失
        if 'weight_loss_disc' in config:
            train_cmd.extend(["--weight_loss_disc", str(config['weight_loss_disc'])])
    elif algo_name == 'MSPAD':
        # MSPAD不使用weight_loss_disc（已由weight_loss_ms_disc替代），不添加该参数
        # 添加多尺度域对抗损失权重（MSPAD特有）
        if 'weight_loss_ms_disc' in config:
            train_cmd.extend(["--weight_loss_ms_disc", str(config['weight_loss_ms_disc'])])
        
        # 添加原型网络间隔参数
        if 'prototypical_margin' in config:
            train_cmd.extend(["--prototypical_margin", str(config['prototypical_margin'])])
        
        # 添加多尺度层权重参数
        if 'scale_weights' in config:
            scale_weights_str = ','.join([str(w) for w in config['scale_weights']])
            train_cmd.extend(["--scale_weights", scale_weights_str])
        
        # 添加层掩码参数
        if 'use_layer_mask' in config:
            layer_mask_str = ','.join([str(m) for m in config['use_layer_mask']])
            train_cmd.extend(["--use_layer_mask", layer_mask_str])
    
    # 运行训练
    print_colored(f"Training {exp_id}...", Colors.BLUE)
    try:
        result = subprocess.run(train_cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print_colored(f"❌ Training failed: {exp_id}", Colors.RED)
            print_colored(result.stderr[:500], Colors.RED)
            return False
    except Exception as e:
        print_colored(f"❌ Training error: {e}", Colors.RED)
        return False
    
    print_colored(f"✓ Training completed: {exp_id}", Colors.GREEN)
    
    # 运行评估
    print_colored(f"Evaluating {exp_id}...", Colors.BLUE)
    eval_cmd = [
        "python", eval_script,
        "--experiments_main_folder", "results",
        "--experiment_folder", exp_folder,
        "--id_src", src,
        "--id_trg", trg,
    ]
    
    try:
        result = subprocess.run(eval_cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print_colored(f"❌ Evaluation failed: {exp_id}", Colors.RED)
            print_colored(result.stderr[:500], Colors.RED)
            return False
    except Exception as e:
        print_colored(f"❌ Evaluation error: {e}", Colors.RED)
        return False
    
    # 保存结果
    save_ablation_result(dataset, src, trg, exp_id, exp_folder, group)

    print_colored(f"✓ Experiment {exp_id} completed successfully", Colors.GREEN)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="MSPAD消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dataset", type=str, default="MSL", choices=["MSL", "SMD", "Boiler", "FWUAV", "ALFA", "UAV"],
                       help="Dataset name (default: MSL)")
    parser.add_argument("--src", type=str, default="F-5",
                       help="Source domain ID (default: F-5 for MSL, 1 for FWUAV)")
    parser.add_argument("--trg", type=str, default=None,
                       help="Target domain ID (default: None, requires --all-targets if not specified)")
    parser.add_argument("--all-targets", action="store_true",
                       help="Use all other files as targets (only set source)")
    parser.add_argument("--group", type=str, choices=["core", "multi_scale", "loss", "all"],
                       default="core", help="Experiment group to run (default: core)")
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="Number of epochs (default: 20)")
    parser.add_argument("--seed", type=int, default=2021,
                       help="Random seed (default: 2021)")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                       help="Skip completed experiments (default: True)")
    
    args = parser.parse_args()

    # 设置数据集特定的默认参数
    if args.dataset == "FWUAV" and args.src == "F-5":
        args.src = "1"  # FWUAV默认使用场景1作为源域
    if args.dataset == "ALFA" and args.src == "F-5":
        args.src = "001"  # ALFA默认使用flight 001作为源域
    if args.dataset == "UAV" and args.src == "F-5":
        args.src = "002"  # UAV默认使用flight 002作为源域

    # 检查参数有效性
    if not args.all_targets and args.trg is None:
        print_colored("Error: Either --trg or --all-targets must be specified", Colors.RED)
        return
    
    # 如果设置了所有目标域
    if args.all_targets:
        print_colored(f"Running with source={args.src}, all other files as targets...", Colors.YELLOW)
        
        files = get_dataset_files(args.dataset)
        if not files:
            print_colored(f"Error: No files found for {args.dataset}", Colors.RED)
            return
        
        if args.src not in files:
            print_colored(f"Error: Source domain '{args.src}' not found in {args.dataset}", Colors.RED)
            return
        
        # 选择要运行的实验组
        if args.group == "all":
            experiments_to_run = {}
            for group_name, group_exps in ABLATION_EXPERIMENTS.items():
                experiments_to_run.update(group_exps)
        else:
            experiments_to_run = ABLATION_EXPERIMENTS.get(args.group, {})
        
        if not experiments_to_run:
            print_colored(f"Error: No experiments found for group '{args.group}'", Colors.RED)
            return
        
        # 打印启动信息
        print_colored("="*80, Colors.CYAN)
        print_colored("MSPAD消融实验", Colors.CYAN)
        print_colored("="*80, Colors.CYAN)
        print_colored(f"Dataset: {args.dataset}", Colors.CYAN)
        print_colored(f"Source: {args.src} -> All other files as targets", Colors.CYAN)
        print_colored(f"Group: {args.group}", Colors.CYAN)
        print_colored(f"Epochs: {args.num_epochs}", Colors.CYAN)
        print_colored(f"Seed: {args.seed}", Colors.CYAN)
        print_colored(f"Total target domains: {len(files) - 1}", Colors.CYAN)
        print_colored("="*80 + "\n", Colors.CYAN)
        
        # 运行所有实验（对所有目标域）
        all_results = {}
        start_time = datetime.now()
        
        for trg in files:
            if trg != args.src:
                print_colored(f"\n{'='*80}", Colors.MAGENTA)
                print_colored(f"Target Domain: {trg}", Colors.MAGENTA)
                print_colored(f"{'='*80}", Colors.MAGENTA)
                
                for exp_id, config in experiments_to_run.items():
                    try:
                        success = run_ablation_experiment(
                            exp_id=exp_id,
                            config=config,
                            dataset=args.dataset,
                            src=args.src,
                            trg=trg,
                            num_epochs=args.num_epochs,
                            seed=args.seed,
                            skip_if_completed=args.skip_completed,
                            group=args.group,
                        )
                        key = f"{exp_id}_{trg}"
                        all_results[key] = {
                            'status': 'Success' if success else 'Failed',
                            'name': config['name'],
                            'description': config['description'],
                            'target': trg,
                        }
                    except Exception as e:
                        print_colored(f"❌ Error in {exp_id} (target: {trg}): {str(e)}", Colors.RED)
                        key = f"{exp_id}_{trg}"
                        all_results[key] = {
                            'status': 'Error',
                            'error': str(e),
                            'target': trg,
                        }
        
        # 保存结果摘要
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration_hours': duration,
            'dataset': args.dataset,
            'source': args.src,
            'targets': 'all',
            'group': args.group,
            'results': all_results,
            'summary': {
                'total': len(all_results),
                'success': sum(1 for r in all_results.values() if r['status'] == 'Success'),
                'failed': sum(1 for r in all_results.values() if r['status'] == 'Failed'),
                'errors': sum(1 for r in all_results.values() if r['status'] == 'Error'),
            }
        }
        
        # 确保结果目录存在
        os.makedirs(ABLATION_DIR, exist_ok=True)
        
        summary_file = os.path.join(ABLATION_DIR, f'ablation_results_{args.dataset}_{args.src}_all_targets.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print_colored("\n" + "="*80, Colors.GREEN)
        print_colored("Experiment Summary", Colors.GREEN)
        print_colored("="*80, Colors.GREEN)
        print_colored(f"Total experiments: {summary['summary']['total']}", Colors.GREEN)
        print_colored(f"Successful: {summary['summary']['success']}", Colors.GREEN)
        print_colored(f"Failed: {summary['summary']['failed']}", Colors.GREEN)
        print_colored(f"Errors: {summary['summary']['errors']}", Colors.GREEN)
        print_colored(f"Duration: {duration:.2f} hours", Colors.GREEN)
        print_colored(f"\nResults saved to: {summary_file}", Colors.GREEN)
        print_colored("="*80 + "\n", Colors.GREEN)
        return
    
    # 如果指定了目标域（原有逻辑）
    # 打印启动信息
    print_colored("="*80, Colors.CYAN)
    print_colored("MSPAD消融实验", Colors.CYAN)
    print_colored("="*80, Colors.CYAN)
    print_colored(f"Dataset: {args.dataset}", Colors.CYAN)
    print_colored(f"Source: {args.src} -> Target: {args.trg}", Colors.CYAN)
    print_colored(f"Group: {args.group}", Colors.CYAN)
    print_colored(f"Epochs: {args.num_epochs}", Colors.CYAN)
    print_colored(f"Seed: {args.seed}", Colors.CYAN)
    print_colored("="*80 + "\n", Colors.CYAN)
    
    # 选择要运行的实验组
    if args.group == "all":
        experiments_to_run = {}
        for group_name, group_exps in ABLATION_EXPERIMENTS.items():
            experiments_to_run.update(group_exps)
    else:
        experiments_to_run = ABLATION_EXPERIMENTS.get(args.group, {})
    
    if not experiments_to_run:
        print_colored(f"Error: No experiments found for group '{args.group}'", Colors.RED)
        return
    
    # 运行所有实验
    results = {}
    start_time = datetime.now()
    
    for exp_id, config in experiments_to_run.items():
        try:
            success = run_ablation_experiment(
                exp_id=exp_id,
                config=config,
                dataset=args.dataset,
                src=args.src,
                trg=args.trg,
                num_epochs=args.num_epochs,
                seed=args.seed,
                skip_if_completed=args.skip_completed,
                group=args.group,
            )
            results[exp_id] = {
                'status': 'Success' if success else 'Failed',
                'name': config['name'],
                'description': config['description'],
            }
        except Exception as e:
            print_colored(f"❌ Error in {exp_id}: {str(e)}", Colors.RED)
            results[exp_id] = {
                'status': 'Error',
                'error': str(e),
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
        'group': args.group,
        'results': results,
        'summary': {
            'total': len(results),
            'success': sum(1 for r in results.values() if r['status'] == 'Success'),
            'failed': sum(1 for r in results.values() if r['status'] == 'Failed'),
            'errors': sum(1 for r in results.values() if r['status'] == 'Error'),
        }
    }
    
    # 确保结果目录存在
    os.makedirs(ABLATION_DIR, exist_ok=True)
    
    summary_file = os.path.join(ABLATION_DIR, f'ablation_results_{args.dataset}_{args.src}_{args.trg}.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print_colored("\n" + "="*80, Colors.GREEN)
    print_colored("Experiment Summary", Colors.GREEN)
    print_colored("="*80, Colors.GREEN)
    print_colored(f"Total experiments: {summary['summary']['total']}", Colors.GREEN)
    print_colored(f"Successful: {summary['summary']['success']}", Colors.GREEN)
    print_colored(f"Failed: {summary['summary']['failed']}", Colors.GREEN)
    print_colored(f"Errors: {summary['summary']['errors']}", Colors.GREEN)
    print_colored(f"Duration: {duration:.2f} hours", Colors.GREEN)
    print_colored(f"\nResults saved to: {summary_file}", Colors.GREEN)
    print_colored("\nDetailed Results:", Colors.GREEN)
    print_colored("-"*80, Colors.GREEN)
    for exp_id, result in results.items():
        status_icon = "✓" if result['status'] == 'Success' else "❌"
        print_colored(f"{status_icon} {exp_id}: {result['status']} - {result.get('name', 'N/A')}", Colors.GREEN)
    print_colored("="*80 + "\n", Colors.GREEN)


if __name__ == '__main__':
    main()

