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
from typing import List, Tuple, Optional
from datetime import datetime
import json

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


def get_dataset_files(dataset: str) -> List[str]:
    """根据数据集名称获取文件列表"""
    if dataset == "MSL":
        return get_msl_files()
    elif dataset == "SMD":
        return get_smd_files()
    elif dataset == "Boiler":
        return get_boiler_files()
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
                "--weight_loss_disc", "0.5",
                "--weight_loss_ms_disc", "0.3",
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
        
        # 结果文件保存在experiment_results文件夹中
        result_path = os.path.join("experiment_results", result_file)
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
        "--num_channels_TCN", dataset_config["num_channels_TCN"],
        "--dilation_factor_TCN", "3",
        "--kernel_size_TCN", "7",
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
            return False
    except Exception as e:
        print_colored(f"❌ Training error: {e}", Colors.RED)
        return False
    
    # 构建评估命令
    eval_cmd = [
        "python", algo_config["eval_script"],
        "--experiments_main_folder", "results",
        "--experiment_folder", exp_folder,
        "--id_src", src,
        "--id_trg", trg,
    ]
    
    # 运行评估
    print_colored(f"\n{'='*60}", Colors.CYAN)
    print_colored(f"  Evaluating: {algo_name} on {dataset} ({src} -> {trg})", Colors.CYAN)
    print_colored(f"{'='*60}", Colors.CYAN)
    
    try:
        result = subprocess.run(eval_cmd, check=False)
        if result.returncode != 0:
            print_colored(f"❌ Evaluation failed: {algo_name} on {dataset} ({src} -> {trg})", Colors.RED)
            return False
    except Exception as e:
        print_colored(f"❌ Evaluation error: {e}", Colors.RED)
        return False
    
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
) -> dict:
    """运行三模型对比实验"""
    
    print_colored("\n" + "="*60, Colors.MAGENTA)
    print_colored(f"Dataset: {dataset}", Colors.MAGENTA)
    print_colored(f"Source: {src} -> Target: {trg}", Colors.MAGENTA)
    print_colored("="*60, Colors.MAGENTA)
    
    algorithms = ["dacad", "MSPAD", "cluda"]
    results = {}
    
    for i, algo in enumerate(algorithms, 1):
        print_colored(f"\n[{i}/3] Running {algo}", Colors.CYAN)
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
        )
        results[algo] = success
    
    # 打印总结
    print_colored("\n" + "="*60, Colors.GREEN)
    print_colored("Comparison Summary:", Colors.GREEN)
    for algo, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        color = Colors.GREEN if success else Colors.RED
        print_colored(f"  {algo:10s}: {status}", color)
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
  
  # 指定源域，所有其他文件为目标域
  python experiments/comparison_experiments.py --dataset MSL --src F-5 --all-targets
  
  # 断点续传：跳过已完成的实验
  python experiments/comparison_experiments.py --dataset SMD --src 1-1 --all-targets --skip-completed
  
  # 运行所有数据集的所有组合
  python experiments/comparison_experiments.py --all-datasets --all-combinations
        """
    )
    
    parser.add_argument("--dataset", type=str, choices=["MSL", "SMD", "Boiler"],
                       help="Dataset name: MSL, SMD, or Boiler")
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
    parser.add_argument("--seed", type=int, default=1234,
                       help="Random seed (default: 1234)")
    parser.add_argument("--skip-completed", action="store_true",
                       help="Skip experiments that are already completed")
    
    args = parser.parse_args()
    
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
    
    # 检查数据集是否设置
    if not args.dataset:
        print_colored("Error: Dataset not specified. Use --dataset MSL|SMD|Boiler", Colors.RED)
        return
    
    # 检查源域是否设置
    if not args.src:
        print_colored("Error: Source domain not specified. Use --src SOURCE_DOMAIN", Colors.RED)
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
        
        for trg in files:
            if trg != args.src:
                results = run_comparison(
                    dataset=args.dataset,
                    src=args.src,
                    trg=trg,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    skip_if_completed=args.skip_completed,
                )
                all_results[f"{args.src}_{trg}"] = results
        
        print_colored("\n" + "="*60, Colors.GREEN)
        print_colored("All experiments completed!", Colors.GREEN)
        print_colored("="*60, Colors.GREEN)
        return
    
    # 如果设置了目标域
    if args.trg:
        run_comparison(
            dataset=args.dataset,
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

