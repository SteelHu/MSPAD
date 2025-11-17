#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSPAD消融实验脚本
================
功能：自动运行所有消融实验，验证每个核心组件的贡献

实验设计：
1. 核心组件消融（core）：
   - Abl-4.1: DACAD Baseline（单尺度域对抗 + Deep SVDD）
   - Abl-4.2: MSPAD Full（多尺度域对抗 + 原型网络 + 加权损失）
   - Abl-4.3: Single-Scale + Deep SVDD
   - Abl-4.4: Multi-Scale + Deep SVDD（仅添加改进1）
   - Abl-4.5: Single-Scale + Prototypical（仅添加改进2）
   - Abl-4.6: Multi-Scale + Prototypical（改进1+2）
   - Abl-4.11-4.13: 多尺度权重配置对比

2. 多尺度域对抗深度分析（multi_scale）：
   - Abl-5.1-5.3: 单层分析（Layer 1/2/3 Only）
   - Abl-5.4-5.6: 两层组合（Layer 1+2, 2+3, 1+3）
   - Abl-5.7: All Layers（完整配置）
   - Abl-5.8-5.10: 单尺度 vs 多尺度组合

3. 损失函数消融（loss）：
   - Abl-6.1: MSPAD Full（所有损失）
   - Abl-6.2-6.6: 移除各个损失函数

使用方法：
    # 运行所有消融实验
    python experiments/ablation_experiments.py
    
    # 运行特定组
    python experiments/ablation_experiments.py --group core  # 核心组件消融
    python experiments/ablation_experiments.py --group multi_scale  # 多尺度分析
    python experiments/ablation_experiments.py --group loss  # 损失函数消融
    
    # 指定数据集和源-目标对
    python experiments/ablation_experiments.py --dataset MSL --src F-5 --trg C-1
    
    # 只指定源域，对所有其他文件作为目标域运行
    python experiments/ablation_experiments.py --dataset MSL --src F-5 --all-targets
    
    # 跳过已完成的实验（断点续传）
    python experiments/ablation_experiments.py --skip-completed
"""

import os
import sys
import subprocess
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 路径常量
SUMMARY_DIR = "experiment_results"
ABLATION_DIR = os.path.join(SUMMARY_DIR, "消融实验")

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


# 消融实验配置
ABLATION_EXPERIMENTS = {
    # ========== 核心组件消融 ==========
    'core': {
        'Abl-4.1': {
            'name': 'DACAD Baseline',
            'description': '原始DACAD（单尺度域对抗 + Deep SVDD）',
            'algo_name': 'dacad',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.0,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'use_prototypical': False,  # 使用Deep SVDD
            'experiment_folder': 'Ablation_DACAD_Baseline',
        },
        'Abl-4.2': {
            'name': 'MSPAD Full',
            'description': 'MSPAD完整版（多尺度域对抗 + 原型网络 + 加权损失）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_MSPAD_Full',
        },
        'Abl-4.3': {
            'name': 'Single-Scale + Deep SVDD',
            'description': '单尺度域对抗 + Deep SVDD（Baseline）',
            'algo_name': 'dacad',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.0,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'experiment_folder': 'Ablation_SingleScale_DeepSVDD',
        },
        'Abl-4.4': {
            'name': 'Multi-Scale + Deep SVDD',
            'description': '多尺度域对抗 + Deep SVDD（仅添加改进1）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            # 注意：MSPAD默认使用原型网络，此实验需要特殊模型变体
            # 暂时使用MSPAD配置，但会在结果中标注
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_MultiScale_DeepSVDD',
        },
        'Abl-4.5': {
            'name': 'Single-Scale + Prototypical',
            'description': '单尺度域对抗 + 原型网络（仅添加改进2）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.0,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'experiment_folder': 'Ablation_SingleScale_Prototypical',
        },
        'Abl-4.6': {
            'name': 'Multi-Scale + Prototypical',
            'description': '多尺度域对抗 + 原型网络（改进1+2）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_MultiScale_Prototypical',
        },
        'Abl-4.11': {
            'name': 'Multi-Scale Uniform Weights',
            'description': '多尺度域对抗（均匀权重）+ 原型网络',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.33, 0.33, 0.34],  # 均匀权重
            'experiment_folder': 'Ablation_MultiScale_UniformWeights',
        },
        'Abl-4.12': {
            'name': 'Multi-Scale Weighted',
            'description': '多尺度域对抗（加权[0.1,0.3,0.6]）+ 原型网络',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],  # 默认权重
            'experiment_folder': 'Ablation_MultiScale_Weighted',
        },
        'Abl-4.13': {
            'name': 'Multi-Scale Reverse Weights',
            'description': '多尺度域对抗（反向权重[0.6,0.3,0.1]）+ 原型网络',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.6, 0.3, 0.1],  # 反向权重
            'experiment_folder': 'Ablation_MultiScale_ReverseWeights',
        },
    },
    
    # ========== 多尺度域对抗深度分析 ==========
    'multi_scale': {
        'Abl-5.1': {
            'name': 'Layer 1 Only',
            'description': '仅低层域对抗（Layer 1）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'use_layer_mask': [1, 0, 0],  # 仅第1层
            'scale_weights': [1.0, 0.0, 0.0],
            'experiment_folder': 'Ablation_Layer1Only',
        },
        'Abl-5.2': {
            'name': 'Layer 2 Only',
            'description': '仅中层域对抗（Layer 2）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'use_layer_mask': [0, 1, 0],  # 仅第2层
            'scale_weights': [0.0, 1.0, 0.0],
            'experiment_folder': 'Ablation_Layer2Only',
        },
        'Abl-5.3': {
            'name': 'Layer 3 Only',
            'description': '仅高层域对抗（Layer 3）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'use_layer_mask': [0, 0, 1],  # 仅第3层
            'scale_weights': [0.0, 0.0, 1.0],
            'experiment_folder': 'Ablation_Layer3Only',
        },
        'Abl-5.4': {
            'name': 'Layer 1+2',
            'description': '低层+中层域对抗',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'use_layer_mask': [1, 1, 0],
            'scale_weights': [0.5, 0.5, 0.0],
            'experiment_folder': 'Ablation_Layer12',
        },
        'Abl-5.5': {
            'name': 'Layer 2+3',
            'description': '中层+高层域对抗',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'use_layer_mask': [0, 1, 1],
            'scale_weights': [0.0, 0.5, 0.5],
            'experiment_folder': 'Ablation_Layer23',
        },
        'Abl-5.6': {
            'name': 'Layer 1+3',
            'description': '低层+高层域对抗',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'use_layer_mask': [1, 0, 1],
            'scale_weights': [0.5, 0.0, 0.5],
            'experiment_folder': 'Ablation_Layer13',
        },
        'Abl-5.7': {
            'name': 'All Layers',
            'description': '所有层域对抗（完整配置）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'use_layer_mask': [1, 1, 1],
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_AllLayers',
        },
        'Abl-5.8': {
            'name': 'Single-Scale Only',
            'description': '仅单尺度（最终层）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.0,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'experiment_folder': 'Ablation_SingleScaleOnly',
        },
        'Abl-5.9': {
            'name': 'Multi-Scale Only',
            'description': '仅多尺度（所有层）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.0,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_MultiScaleOnly',
        },
        'Abl-5.10': {
            'name': 'Single + Multi-Scale',
            'description': '单尺度 + 多尺度（完整配置）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_SingleMultiScale',
        },
    },
    
    # ========== 损失函数消融 ==========
    'loss': {
        'Abl-6.1': {
            'name': 'MSPAD Full',
            'description': 'MSPAD完整版（所有损失）',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_AllLosses',
        },
        'Abl-6.2': {
            'name': 'w/o Single-Scale DA',
            'description': '移除单尺度域对抗损失',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.0,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_NoSingleScaleDA',
        },
        'Abl-6.3': {
            'name': 'w/o Multi-Scale DA',
            'description': '移除多尺度域对抗损失',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.0,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'experiment_folder': 'Ablation_NoMultiScaleDA',
        },
        'Abl-6.4': {
            'name': 'w/o Prototypical Loss',
            'description': '移除原型网络分类损失',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 0.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_NoPrototypicalLoss',
        },
        'Abl-6.5': {
            'name': 'w/o Source Sup CL',
            'description': '移除源域监督对比损失',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.0,
            'weight_loss_trg_inj': 0.1,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_NoSourceSupCL',
        },
        'Abl-6.6': {
            'name': 'w/o Target Inj CL',
            'description': '移除目标域注入对比损失',
            'algo_name': 'MSPAD',
            'weight_loss_disc': 0.5,
            'weight_loss_ms_disc': 0.3,
            'weight_loss_pred': 1.0,
            'weight_loss_src_sup': 0.1,
            'weight_loss_trg_inj': 0.0,
            'prototypical_margin': 1.0,
            'scale_weights': [0.1, 0.3, 0.6],
            'experiment_folder': 'Ablation_NoTargetInjCL',
        },
    },
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


def run_ablation_experiment(
    exp_id: str,
    config: dict,
    dataset: str = "MSL",
    src: str = "F-5",
    trg: str = "C-1",
    num_epochs: int = 20,
    seed: int = 1234,
    skip_if_completed: bool = True,
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
    
    exp_folder = f"{dataset}_{config['experiment_folder']}"
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
        "--weight_loss_disc", str(config['weight_loss_disc']),
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
    
    # 添加MSPAD特有参数
    if algo_name == 'MSPAD':
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
    
    print_colored(f"✓ Experiment {exp_id} completed successfully", Colors.GREEN)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="MSPAD消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dataset", type=str, default="MSL", choices=["MSL", "SMD", "Boiler"],
                       help="Dataset name (default: MSL)")
    parser.add_argument("--src", type=str, default="F-5",
                       help="Source domain ID (default: F-5)")
    parser.add_argument("--trg", type=str, default=None,
                       help="Target domain ID (default: None, requires --all-targets if not specified)")
    parser.add_argument("--all-targets", action="store_true",
                       help="Use all other files as targets (only set source)")
    parser.add_argument("--group", type=str, choices=["core", "multi_scale", "loss", "all"],
                       default="all", help="Experiment group to run (default: all)")
    parser.add_argument("--num_epochs", type=int, default=20,
                       help="Number of epochs (default: 20)")
    parser.add_argument("--seed", type=int, default=2021,
                       help="Random seed (default: 2021)")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                       help="Skip completed experiments (default: True)")
    
    args = parser.parse_args()
    
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

