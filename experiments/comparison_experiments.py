#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSPAD对比实验脚本
================
功能：在MSL、SMD、Boiler数据集上对比MSPAD与SOTA方法（DACAD、CLUDA）的性能

使用方法：
    # 单个数据集，单个源-目标对
    python experiments/comparison_experiments.py --dataset MSL --src F-5 --trg C-1
    
    # 单个数据集，指定源域，所有其他为目标域
    python experiments/comparison_experiments.py --dataset MSL --src F-5 --all-targets
    
    # 所有数据集的所有组合
    python experiments/comparison_experiments.py --all-datasets --all-combinations
    
    # 断点续传
    python experiments/comparison_experiments.py --dataset MSL --src F-5 --all-targets --skip-completed
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Optional
from datetime import datetime
import json
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


def get_alfa_files() -> List[str]:
    """获取ALFA数据集的所有文件列表"""
    alfa_dir = 'datasets/ALFA'
    if not os.path.exists(alfa_dir):
        print_colored(f"Error: ALFA dataset directory not found: {alfa_dir}", Colors.RED)
        return []

    files = []
    for item in os.listdir(alfa_dir):
        item_path = os.path.join(alfa_dir, item)
        # ALFA数据集的目录结构是直接的数字编号（如002、003等）
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
    elif dataset == "ALFA":
        return get_alfa_files()
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


def get_algorithm_config(algo_name: str) -> dict:
    """获取算法的配置参数"""
    configs = {
        "dacad": {
            "train_script": "main/train.py",
            "eval_script": "main/eval.py",
            "exp_folder_suffix": "Baseline_DACAD",
            "result_file_prefix": "DACAD_",
            "extra_params": [
                "--weight_loss_disc", "0.5",
                "--weight_loss_pred", "1.0",
                "--weight_loss_src_sup", "0.1",
                "--weight_loss_trg_inj", "0.1",
            ],
        },
        "MSPAD": {
            "train_script": "main_new/train.py",
            "eval_script": "main_new/eval.py",
            "exp_folder_suffix": "MSPAD_Full",
            "result_file_prefix": "MSPAD_",
            "extra_params": [
                "--weight_loss_disc", "0.0",
                "--weight_loss_ms_disc", "0.5",
                "--prototypical_margin", "1.0",
                "--weight_loss_pred", "1.0",
                "--weight_loss_src_sup", "0.1",
                "--weight_loss_trg_inj", "0.1",
            ],
        },
        "cluda": {
            "train_script": "main_cluda/train.py",
            "eval_script": "main_cluda/eval.py",
            "exp_folder_suffix": "cluda",
            "result_file_prefix": "CLUDA_",
            "batch_size": 128,  # CLUDA需要与数据集兼容的batch_size
            "num_channels_TCN": "64-64-64-64-64",  # CLUDA使用自己的网络配置
            "kernel_size_TCN": 3,  # CLUDA使用自己的卷积核大小
            "extra_params": [
                "--weight_loss_disc", "1.0",
                "--weight_loss_pred", "1.0",
                "--weight_loss_src", "1.0",
                "--weight_loss_trg", "1.0",
                "--weight_loss_ts", "1.0",
                "--use_mask",
                "--num_steps", "1000",
                "--checkpoint_freq", "100",
                "--weight_decay", "1e-2",
            ],
        },
    }
    return configs.get(algo_name, {})


def is_experiment_completed(
    dataset: str,
    algo_name: str,
    src: str,
    trg: str,
    check_model_file: bool = True,
    check_csv_file: bool = True,
) -> bool:
    """检查实验是否已完成"""
    algo_config = get_algorithm_config(algo_name)
    if not algo_config:
        return False
    
    exp_folder = f"{dataset}_{algo_config['exp_folder_suffix']}"
    exp_dir = os.path.join("results", exp_folder, f"{src}-{trg}")
    
    # 方法1: 检查模型文件是否存在
    if check_model_file:
        model_file = os.path.join(exp_dir, "model_best.pth.tar")
        if os.path.exists(model_file):
            pred_file = os.path.join(exp_dir, "predictions_test_target.csv")
            if os.path.exists(pred_file):
                return True
    
    # 方法2: 检查CSV结果文件中是否已有该组合
    if check_csv_file:
        dataset_lower = dataset.lower()
        if algo_name == "cluda":
            result_file = f"CLUDA_{dataset}_{src}.csv"
        elif algo_name == "dacad":
            result_file = f"DACAD_{dataset}_{src}.csv"
        elif algo_name == "MSPAD":
            result_file = f"MSPAD_{dataset}_{src}.csv"
        else:
            result_file = f"{algo_name}_{dataset}_{src}.csv"
        
        # 结果文件保存在experiment_results/对比实验文件夹中
        result_path = os.path.join("experiment_results", "对比实验", result_file)
        if os.path.exists(result_path):
            try:
                df = pd.read_csv(result_path)
                if 'src_id' in df.columns and 'trg_id' in df.columns:
                    completed = df[(df['src_id'] == src) & (df['trg_id'] == trg)]
                    if len(completed) > 0:
                        return True
            except Exception:
                pass
    
    return False


def save_comparison_result(
    dataset: str,
    algo_name: str,
    src: str,
    trg: str,
    exp_folder: str,
) -> bool:
    """保存对比实验结果到CSV文件"""
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

        # 准备结果数据
        result_data = {
            'dataset': dataset,
            'src_trg': actual_src_trg,  # 使用实际的源-目标对
            'algorithm': algo_name,
            'exp_folder': exp_folder,
            'AUROC': auroc if auroc is not None else float('nan'),
            'AUPRC': auprc if auprc is not None else float('nan'),
            'Best_F1': best_f1 if best_f1 is not None else float('nan'),
        }

        # 保存到对比实验文件夹
        comparison_dir = os.path.join("experiment_results", "对比实验")
        os.makedirs(comparison_dir, exist_ok=True)
        comparison_csv = os.path.join(comparison_dir, f'Comparison_{dataset}_{src}_{trg}.csv')

        # 追加或创建文件
        mode = 'a' if os.path.exists(comparison_csv) else 'w'
        header = mode == 'w'

        df = pd.DataFrame([result_data])
        df.to_csv(comparison_csv, mode=mode, header=header, index=False)

        print_colored(f"✓ 结果已保存到: {comparison_csv}", Colors.GREEN)
        return True
    except Exception as e:
        print_colored(f"⚠ Warning: Failed to save comparison results: {e}", Colors.YELLOW)
        return False


def run_experiment(
    dataset: str,
    algo_name: str,
    src: str,
    trg: str,
    num_epochs: int = 20,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-4,
    seed: int = 1234,
    skip_if_completed: bool = False,
    continue_on_error: bool = False,
) -> bool:
    """运行单个实验"""
    
    if skip_if_completed:
        if is_experiment_completed(dataset, algo_name, src, trg):
            print_colored(f"⏭  Skipped (already completed): {algo_name} on {dataset} ({src} -> {trg})", Colors.YELLOW)
            return True
    
    dataset_config = get_dataset_config(dataset)
    if not dataset_config:
        print_colored(f"Unknown dataset: {dataset}", Colors.RED)
        return False
    
    algo_config = get_algorithm_config(algo_name)
    if not algo_config:
        print_colored(f"Unknown algorithm: {algo_name}", Colors.RED)
        return False
    
    # 优先使用算法特定的batch_size，如果没有则使用数据集默认的或函数参数指定的
    if "batch_size" in algo_config:
        final_batch_size = algo_config["batch_size"]
    else:
        final_batch_size = batch_size if batch_size is not None else dataset_config["batch_size"]
    exp_folder = f"{dataset}_{algo_config['exp_folder_suffix']}"
    
    # 构建训练命令
    train_cmd = [
        "python", algo_config["train_script"],
        "--algo_name", algo_name,
        "--num_epochs", str(num_epochs),
        "--batch_size", str(final_batch_size),
        "--eval_batch_size", str(final_batch_size),
        "--learning_rate", str(learning_rate),
        "--dropout", str(dataset_config["dropout"]),
        "--num_channels_TCN", algo_config.get("num_channels_TCN", dataset_config["num_channels_TCN"]),
        "--dilation_factor_TCN", "3",
        "--kernel_size_TCN", str(algo_config.get("kernel_size_TCN", 7)),
        "--hidden_dim_MLP", str(dataset_config["hidden_dim_MLP"]),
        "--queue_size", "98304",
        "--momentum", "0.99",
        "--id_src", src,
        "--id_trg", trg,
        "--path_src", dataset_config["path_src"],
        "--path_trg", dataset_config["path_trg"],
        "--experiments_main_folder", "results",
        "--experiment_folder", exp_folder,
        "--seed", str(seed),
    ]
    
    train_cmd.extend(algo_config["extra_params"])
    
    # 运行训练
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"  Training: {algo_name} on {dataset} ({src} -> {trg})", Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)

    try:
        result = subprocess.run(train_cmd, check=False)
        if result.returncode != 0:
            print_colored(f"❌ Training failed: {algo_name} on {dataset} ({src} -> {trg})", Colors.RED)
            if continue_on_error:
                print_colored(f"⚠ Continuing with other algorithms despite training failure", Colors.YELLOW)
                return False
            return False
    except Exception as e:
        print_colored(f"❌ Training error: {e}", Colors.RED)
        if continue_on_error:
            print_colored(f"⚠ Continuing with other algorithms despite training error", Colors.YELLOW)
            return False
        return False
    
    # 构建评估命令
    eval_cmd = [
        "python", algo_config["eval_script"],
        "--experiments_main_folder", "results",
        "--experiment_folder", exp_folder,
        "--id_src", src,
        "--id_trg", trg,
    ]

    # 所有算法默认都会删除模型文件以节省空间
    
    # 运行评估
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"  Evaluating: {algo_name} on {dataset} ({src} -> {trg})", Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)

    try:
        result = subprocess.run(eval_cmd, check=False)
        if result.returncode != 0:
            print_colored(f"❌ Evaluation failed: {algo_name} on {dataset} ({src} -> {trg})", Colors.RED)
            if continue_on_error:
                print_colored(f"⚠ Continuing with other algorithms despite evaluation failure", Colors.YELLOW)
                return False
            return False
    except Exception as e:
        print_colored(f"❌ Evaluation error: {e}", Colors.RED)
        if continue_on_error:
            print_colored(f"⚠ Continuing with other algorithms despite evaluation error", Colors.YELLOW)
            return False
        return False
    
    # 删除模型文件以节省空间
    try:
        model_file = os.path.join("results", exp_folder, f"{src}-{trg}", "model_best.pth.tar")
        if os.path.exists(model_file):
            os.remove(model_file)
            print_colored(f"🗑️  Model file deleted: {model_file}", Colors.BLUE)
    except Exception as e:
        print_colored(f"⚠ Warning: Failed to delete model file: {e}", Colors.YELLOW)

    # 保存结果
    save_comparison_result(dataset, algo_name, src, trg, exp_folder)

    print_colored(f"✓ Completed: {algo_name} on {dataset} ({src} -> {trg})", Colors.GREEN)
    return True


def run_comparison(
    dataset: str,
    src: str,
    trg: str,
    num_epochs: int = 20,
    batch_size: Optional[int] = None,
    learning_rate: float = 1e-4,
    seed: int = 1234,
    skip_if_completed: bool = False,
    continue_on_error: bool = True,
) -> dict:
    """运行三模型对比实验"""
    
    print_colored("\n" + "="*60, Colors.MAGENTA)
    print_colored(f"Dataset: {dataset}", Colors.MAGENTA)
    print_colored(f"Source: {src} -> Target: {trg}", Colors.MAGENTA)
    print_colored("="*60, Colors.MAGENTA)
    
    algorithms = ["dacad", "MSPAD", "cluda"]

    results = {}

    for i, algo in enumerate(algorithms, 1):
        print_colored(f"\n[{i}/{len(algorithms)}] Running {algo}", Colors.CYAN)
        success = run_experiment(
            dataset=dataset,
            algo_name=algo,
            src=src,
            trg=trg,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            skip_if_completed=skip_if_completed,
            continue_on_error=continue_on_error,
        )
        results[algo] = success
    
    # 打印总结
    print_colored("\n" + "="*60, Colors.GREEN)
    print_colored("Comparison Summary:", Colors.GREEN)
    successful_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    for algo, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        color = Colors.GREEN if success else Colors.RED
        print_colored(f"  {algo:10s}: {status}", color)

    if successful_count == total_count:
        print_colored(f"All {total_count} algorithms completed successfully!", Colors.GREEN)
    elif successful_count > 0:
        print_colored(f"{successful_count}/{total_count} algorithms completed successfully.", Colors.YELLOW)
    else:
        print_colored(f"All {total_count} algorithms failed.", Colors.RED)

    print_colored("="*60 + "\n", Colors.GREEN)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MSPAD对比实验：与DACAD和CLUDA对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单个实验
  python experiments/comparison_experiments.py --dataset MSL --src F-5 --trg C-1
  python experiments/comparison_experiments.py --dataset FWUAV --src 1 --trg 6

  # 指定源域，所有其他文件为目标域
  python experiments/comparison_experiments.py --dataset MSL --src F-5 --all-targets
  python experiments/comparison_experiments.py --dataset FWUAV --src 1 --all-targets
  python experiments/comparison_experiments.py --dataset UAV --src 002 --all-targets

  # 多个数据集，指定源域，所有其他文件为目标域
  python experiments/comparison_experiments.py --datasets ALFA FWUAV --src 001 --all-targets
  python experiments/comparison_experiments.py --datasets FWUAV ALFA --src 1 --all-targets
  python experiments/comparison_experiments.py --datasets UAV ALFA --src 002 --all-targets

  # 断点续传：跳过已完成的实验
  python experiments/comparison_experiments.py --dataset SMD --src 1-1 --all-targets --skip-completed
  python experiments/comparison_experiments.py --datasets ALFA FWUAV --src 001 --all-targets --skip-completed

  # 运行所有数据集的所有组合
  python experiments/comparison_experiments.py --all-datasets --all-combinations
        """
    )
    
    parser.add_argument("--dataset", type=str, choices=["MSL", "SMD", "Boiler", "FWUAV", "ALFA", "UAV"],
                       help="Dataset name: MSL, SMD, Boiler, FWUAV, ALFA, or UAV")
    parser.add_argument("--datasets", type=str, nargs='+',
                       choices=["MSL", "SMD", "Boiler", "FWUAV", "ALFA", "UAV"],
                       help="Multiple dataset names (space-separated)")
    parser.add_argument("--src", type=str, help="Source domain ID")
    parser.add_argument("--trg", type=str, help="Target domain ID")
    parser.add_argument("--all-targets", action="store_true",
                       help="Use all other files as targets (only set source)")
    parser.add_argument("--all-datasets", action="store_true",
                       help="Run on all datasets")
    parser.add_argument("--all-combinations", action="store_true",
                       help="Run all source-target combinations")
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="Number of epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (default: dataset-specific)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--skip-completed", action="store_true",
                       help="Skip experiments that are already completed")
    
    args = parser.parse_args()

    # 设置数据集特定的默认参数
    if args.dataset == "FWUAV" and not args.src:
        args.src = "1"  # FWUAV默认使用场景1作为源域
    if args.dataset == "ALFA" and not args.src:
        args.src = "001"  # ALFA默认使用flight 001作为源域
    if args.dataset == "UAV" and not args.src:
        args.src = "002"  # UAV默认使用flight 002作为源域

    # 打印启动信息
    print_colored("="*60, Colors.CYAN)
    print_colored("MSPAD对比实验：与DACAD和CLUDA对比", Colors.CYAN)
    print_colored("="*60, Colors.CYAN)
    print_colored(f"Parameters:", Colors.CYAN)
    print_colored(f"  Epochs: {args.num_epochs}", Colors.CYAN)
    print_colored(f"  Batch Size: {args.batch_size if args.batch_size else 'dataset-specific'}", Colors.CYAN)
    print_colored(f"  Learning Rate: {args.learning_rate}", Colors.CYAN)
    print_colored(f"  Seed: {args.seed}", Colors.CYAN)
    if args.skip_completed:
        print_colored(f"  Skip Completed: ✓ (断点续传模式)", Colors.YELLOW)
    print_colored("="*60 + "\n", Colors.CYAN)
    
    all_results = {}
    start_time = datetime.now()
    
    # 如果设置了所有数据集和所有组合
    if args.all_datasets and args.all_combinations:
        print_colored("Running all datasets with all combinations...", Colors.YELLOW)
        
        datasets = ["MSL", "SMD", "Boiler"]
        for dataset in datasets:
            print_colored(f"\n{'='*60}", Colors.BLUE)
            print_colored(f"=== {dataset} Dataset ===", Colors.BLUE)
            print_colored(f"{'='*60}", Colors.BLUE)
            
            files = get_dataset_files(dataset)
            if not files:
                print_colored(f"⚠ No files found for {dataset}, skipping...", Colors.YELLOW)
                continue
            
            for src in files:
                for trg in files:
                    if src != trg:
                        results = run_comparison(
                            dataset=dataset,
                            src=src,
                            trg=trg,
                            num_epochs=args.num_epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate,
                            seed=args.seed,
                            skip_if_completed=args.skip_completed,
                        )
                        all_results[f"{dataset}_{src}_{trg}"] = results
        
        print_colored("\n" + "="*60, Colors.GREEN)
        print_colored("All experiments completed!", Colors.GREEN)
        print_colored("="*60, Colors.GREEN)
        
        # 保存结果摘要
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration_hours': duration,
            'results': all_results,
        }
        summary_file = 'comparison_experiments_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print_colored(f"\nResults saved to: {summary_file}", Colors.GREEN)
        return
    
    # 确定要运行的数据集列表
    if args.datasets:
        datasets_to_run = args.datasets
    elif args.dataset:
        datasets_to_run = [args.dataset]
    elif args.all_datasets:
        datasets_to_run = ["MSL", "SMD", "Boiler", "FWUAV", "ALFA", "UAV"]
    else:
        print_colored("Error: Dataset not specified. Use --dataset, --datasets, or --all-datasets", Colors.RED)
        return

    # 检查源域是否设置
    if not args.src:
        print_colored("Error: Source domain not specified. Use --src SOURCE_DOMAIN", Colors.RED)
        return
    
    # 如果设置了所有目标域
    if args.all_targets:
        print_colored(f"Running with source={args.src}, all other files as targets...", Colors.YELLOW)

        # 为每个数据集执行
        for dataset in datasets_to_run:
            print_colored(f"\n{'='*60}", Colors.BLUE)
            print_colored(f"=== {dataset} Dataset ===", Colors.BLUE)
            print_colored(f"{'='*60}", Colors.BLUE)

            # 获取数据集的文件列表
            files = get_dataset_files(dataset)
            if not files:
                print_colored(f"⚠ No files found for {dataset}, skipping...", Colors.YELLOW)
                continue

            # 检查源域是否存在
            if args.src not in files:
                print_colored(f"⚠ Source domain '{args.src}' not found in {dataset}, skipping...", Colors.YELLOW)
                continue

            print_colored(f"Found {len(files)} files, using {args.src} as source", Colors.CYAN)

            # 为每个目标域运行实验
            for trg in files:
                if trg != args.src:
                    results = run_comparison(
                        dataset=dataset,
                        src=args.src,
                        trg=trg,
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        seed=args.seed,
                        skip_if_completed=args.skip_completed,
                    )
                    all_results[f"{dataset}_{args.src}_{trg}"] = results

        print_colored("\n" + "="*60, Colors.GREEN)
        print_colored("All experiments completed!", Colors.GREEN)
        print_colored("="*60, Colors.GREEN)
        return
    
    # 如果设置了目标域
    if args.trg:
        if len(datasets_to_run) > 1:
            print_colored("Running single target experiment on multiple datasets...", Colors.YELLOW)
            for dataset in datasets_to_run:
                print_colored(f"\n{'='*60}", Colors.BLUE)
                print_colored(f"=== {dataset} Dataset ===", Colors.BLUE)
                print_colored(f"{'='*60}", Colors.BLUE)

                run_comparison(
                    dataset=dataset,
                    src=args.src,
                    trg=args.trg,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    skip_if_completed=args.skip_completed,
                )
        else:
            run_comparison(
                dataset=datasets_to_run[0],
                src=args.src,
                trg=args.trg,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                skip_if_completed=args.skip_completed,
            )
        return
    
    # 如果没有设置目标域，也没有设置--all-targets
    print_colored("Error: Target domain not specified. Use --trg TARGET_DOMAIN or --all-targets", Colors.RED)


if __name__ == "__main__":
    main()

