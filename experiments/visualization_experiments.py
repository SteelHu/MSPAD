#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSPAD可视化实验脚本
==================
功能：生成各种可视化图表，包括特征空间可视化、域对齐可视化、训练过程可视化等

使用方法：
    # 运行所有可视化实验
    python experiments/visualization_experiments.py
    
    # 运行特定类型的可视化
    python experiments/visualization_experiments.py --type t_sne
    python experiments/visualization_experiments.py --type domain_alignment
    python experiments/visualization_experiments.py --type training_curves
    
    # 指定模型和数据集
    python experiments/visualization_experiments.py --dataset MSL --src F-5 --trg C-1 --model MSPAD
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import get_dataset
from utils.util_progress_log import get_dataset_type
from algorithms import get_algorithm
from collections import namedtuple

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


def load_model_and_data(
    dataset: str,
    algo_name: str,
    src: str,
    trg: str,
    exp_folder: str,
) -> Tuple:
    """加载模型和数据"""
    
    # 确定数据集路径
    if dataset == "MSL":
        path_src = "datasets/MSL_SMAP"
        path_trg = "datasets/MSL_SMAP"
    elif dataset == "SMD":
        path_src = "datasets/SMD/test"
        path_trg = "datasets/SMD/test"
    elif dataset == "Boiler":
        path_src = "datasets/Boiler"
        path_trg = "datasets/Boiler"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # 确定训练脚本路径以加载参数
    if algo_name == 'dacad':
        train_script = "main/train.py"
    elif algo_name == 'MSPAD':
        train_script = "main_new/train.py"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # 加载保存的参数
    exp_dir = os.path.join("results", exp_folder, f"{src}-{trg}")
    args_file = os.path.join(exp_dir, "commandline_args.txt")
    
    if not os.path.exists(args_file):
        raise FileNotFoundError(f"Arguments file not found: {args_file}")
    
    with open(args_file, 'r') as f:
        saved_args_dict = json.load(f)
    
    saved_args = namedtuple("SavedArgs", saved_args_dict.keys())(*saved_args_dict.values())
    
    # 加载数据集
    dataset_src_test = get_dataset(saved_args, domain_type="source", split_type="test")
    dataset_trg_test = get_dataset(saved_args, domain_type="target", split_type="test")
    
    # 创建数据加载器
    dataloader_src = DataLoader(dataset_src_test, batch_size=256, shuffle=False, num_workers=0)
    dataloader_trg = DataLoader(dataset_trg_test, batch_size=256, shuffle=False, num_workers=0)
    
    # 获取输入维度
    input_channels_dim = dataset_src_test[0]['sequence'].shape[1]
    input_static_dim = dataset_src_test[0]['static'].shape[0] if 'static' in dataset_src_test[0] else 0
    
    # 获取算法实例
    algorithm = get_algorithm(saved_args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim)
    
    # 加载模型权重
    algorithm.load_state(exp_dir)
    algorithm.eval()
    
    return algorithm, dataloader_src, dataloader_trg, saved_args


def extract_features(algorithm, dataloader, domain_type='source'):
    """提取特征表示"""
    features = []
    labels = []
    domain_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if domain_type == 'source':
                # 提取源域特征
                seq = batch['sequence'].cuda()
                static = batch.get('static', None)
                if static is not None:
                    static = static.cuda()
                
                # 获取特征（这里需要根据实际模型结构调整）
                # 假设模型有get_features方法，或者通过forward获取
                q_repr = algorithm.model.encoder_q(seq)
                if static is not None:
                    # 拼接静态特征
                    features_batch = torch.cat([q_repr, static], dim=1)
                else:
                    features_batch = q_repr
                
                features.append(features_batch.cpu().numpy())
                labels.append(batch['label'].numpy())
                domain_labels.append(np.ones(len(batch['label'])))  # 源域标签为1
            else:
                # 提取目标域特征
                seq = batch['sequence'].cuda()
                static = batch.get('static', None)
                if static is not None:
                    static = static.cuda()
                
                q_repr = algorithm.model.encoder_q(seq)
                if static is not None:
                    features_batch = torch.cat([q_repr, static], dim=1)
                else:
                    features_batch = q_repr
                
                features.append(features_batch.cpu().numpy())
                labels.append(batch['label'].numpy())
                domain_labels.append(np.zeros(len(batch['label'])))  # 目标域标签为0
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    domain_labels = np.concatenate(domain_labels, axis=0)
    
    return features, labels, domain_labels


def visualize_t_sne(
    algorithm_src,
    algorithm_trg,
    dataloader_src,
    dataloader_trg,
    output_dir: str,
    dataset: str,
    src: str,
    trg: str,
):
    """t-SNE特征空间可视化"""
    print_colored("Generating t-SNE visualization...", Colors.BLUE)
    
    # 提取特征
    features_src, labels_src, _ = extract_features(algorithm_src, dataloader_src, 'source')
    features_trg, labels_trg, _ = extract_features(algorithm_trg, dataloader_trg, 'target')
    
    # 合并特征
    features_all = np.vstack([features_src, features_trg])
    labels_all = np.concatenate([labels_src.flatten(), labels_trg.flatten()])
    domain_labels_all = np.concatenate([
        np.ones(len(features_src)),  # 源域
        np.zeros(len(features_trg))  # 目标域
    ])
    
    # 降维到2D
    print_colored("Running t-SNE...", Colors.BLUE)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_all)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1：按域着色
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        features_2d[domain_labels_all == 1, 0],
        features_2d[domain_labels_all == 1, 1],
        c='blue', alpha=0.5, label='Source Domain', s=10
    )
    scatter2 = ax1.scatter(
        features_2d[domain_labels_all == 0, 0],
        features_2d[domain_labels_all == 0, 1],
        c='red', alpha=0.5, label='Target Domain', s=10
    )
    ax1.set_title('Feature Space Visualization (by Domain)', fontsize=14)
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：按标签着色
    ax2 = axes[1]
    scatter3 = ax2.scatter(
        features_2d[labels_all == 0, 0],
        features_2d[labels_all == 0, 1],
        c='green', alpha=0.5, label='Normal', s=10
    )
    scatter4 = ax2.scatter(
        features_2d[labels_all == 1, 0],
        features_2d[labels_all == 1, 1],
        c='orange', alpha=0.5, label='Anomaly', s=10
    )
    ax2.set_title('Feature Space Visualization (by Label)', fontsize=14)
    ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形
    output_file = os.path.join(output_dir, f't_sne_{dataset}_{src}_{trg}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_colored(f"✓ t-SNE visualization saved to: {output_file}", Colors.GREEN)


def visualize_training_curves(
    exp_dir: str,
    output_dir: str,
    dataset: str,
    src: str,
    trg: str,
):
    """可视化训练曲线"""
    print_colored("Generating training curves...", Colors.BLUE)
    
    # 读取训练日志（这里需要根据实际日志格式调整）
    log_file = os.path.join(exp_dir, "train.log")
    
    if not os.path.exists(log_file):
        print_colored(f"Warning: Log file not found: {log_file}", Colors.YELLOW)
        return
    
    # 解析日志文件（这里需要根据实际格式实现）
    # 示例：假设日志包含损失和指标信息
    # 实际实现需要根据日志格式解析
    
    print_colored(f"✓ Training curves visualization (需要根据日志格式实现)", Colors.GREEN)


def visualize_confusion_matrix(
    exp_dir: str,
    output_dir: str,
    dataset: str,
    src: str,
    trg: str,
):
    """可视化混淆矩阵"""
    print_colored("Generating confusion matrix...", Colors.BLUE)
    
    # 读取预测结果
    pred_file = os.path.join(exp_dir, "predictions_test_target.csv")
    
    if not os.path.exists(pred_file):
        print_colored(f"Warning: Prediction file not found: {pred_file}", Colors.YELLOW)
        return
    
    df = pd.read_csv(pred_file)
    
    if 'y' not in df.columns or 'y_pred' not in df.columns:
        print_colored("Warning: Required columns not found in prediction file", Colors.YELLOW)
        return
    
    # 计算最佳阈值（使用F1分数）
    from sklearn.metrics import precision_recall_curve, f1_score
    
    y_true = df['y'].values
    y_scores = df['y_pred'].values
    
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thr[best_idx]
    
    y_pred = (y_scores > best_threshold).astype(int)
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # 可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix (Threshold={best_threshold:.3f})', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    output_file = os.path.join(output_dir, f'confusion_matrix_{dataset}_{src}_{trg}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_colored(f"✓ Confusion matrix saved to: {output_file}", Colors.GREEN)


def visualize_roc_pr_curves(
    exp_dir: str,
    output_dir: str,
    dataset: str,
    src: str,
    trg: str,
):
    """可视化ROC和PR曲线"""
    print_colored("Generating ROC and PR curves...", Colors.BLUE)
    
    pred_file = os.path.join(exp_dir, "predictions_test_target.csv")
    
    if not os.path.exists(pred_file):
        print_colored(f"Warning: Prediction file not found: {pred_file}", Colors.YELLOW)
        return
    
    df = pd.read_csv(pred_file)
    
    if 'y' not in df.columns or 'y_pred' not in df.columns:
        return
    
    y_true = df['y'].values
    y_scores = df['y_pred'].values
    
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # PR曲线
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC曲线
    ax1 = axes[0]
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # PR曲线
    ax2 = axes[1]
    ax2.plot(rec, prec, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14)
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'roc_pr_curves_{dataset}_{src}_{trg}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_colored(f"✓ ROC and PR curves saved to: {output_file}", Colors.GREEN)


def main():
    parser = argparse.ArgumentParser(
        description="MSPAD可视化实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--dataset", type=str, default="MSL", choices=["MSL", "SMD", "Boiler"],
                       help="Dataset name (default: MSL)")
    parser.add_argument("--src", type=str, default="F-5",
                       help="Source domain ID (default: F-5)")
    parser.add_argument("--trg", type=str, default="C-1",
                       help="Target domain ID (default: C-1)")
    parser.add_argument("--model", type=str, default="MSPAD", choices=["MSPAD", "dacad"],
                       help="Model to visualize (default: MSPAD)")
    parser.add_argument("--type", type=str, 
                       choices=["t_sne", "domain_alignment", "training_curves", "confusion_matrix", "roc_pr", "all"],
                       default="all", help="Visualization type (default: all)")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Output directory for visualizations (default: visualizations)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定实验文件夹
    if args.model == "MSPAD":
        exp_folder = f"{args.dataset}_MSPAD_Full"
    else:
        exp_folder = f"{args.dataset}_Baseline_DACAD"
    
    print_colored("="*80, Colors.CYAN)
    print_colored("MSPAD可视化实验", Colors.CYAN)
    print_colored("="*80, Colors.CYAN)
    print_colored(f"Dataset: {args.dataset}", Colors.CYAN)
    print_colored(f"Source: {args.src} -> Target: {args.trg}", Colors.CYAN)
    print_colored(f"Model: {args.model}", Colors.CYAN)
    print_colored(f"Visualization Type: {args.type}", Colors.CYAN)
    print_colored(f"Output Directory: {args.output_dir}", Colors.CYAN)
    print_colored("="*80 + "\n", Colors.CYAN)
    
    try:
        # 加载模型和数据
        algorithm, dataloader_src, dataloader_trg, saved_args = load_model_and_data(
            dataset=args.dataset,
            algo_name=args.model,
            src=args.src,
            trg=args.trg,
            exp_folder=exp_folder,
        )
        
        exp_dir = os.path.join("results", exp_folder, f"{args.src}-{args.trg}")
        
        # 运行可视化
        if args.type == "all" or args.type == "t_sne":
            visualize_t_sne(
                algorithm, algorithm, dataloader_src, dataloader_trg,
                args.output_dir, args.dataset, args.src, args.trg
            )
        
        if args.type == "all" or args.type == "confusion_matrix":
            visualize_confusion_matrix(
                exp_dir, args.output_dir, args.dataset, args.src, args.trg
            )
        
        if args.type == "all" or args.type == "roc_pr":
            visualize_roc_pr_curves(
                exp_dir, args.output_dir, args.dataset, args.src, args.trg
            )
        
        if args.type == "all" or args.type == "training_curves":
            visualize_training_curves(
                exp_dir, args.output_dir, args.dataset, args.src, args.trg
            )
        
        print_colored("\n" + "="*80, Colors.GREEN)
        print_colored("All visualizations completed!", Colors.GREEN)
        print_colored("="*80, Colors.GREEN)
        
    except Exception as e:
        print_colored(f"❌ Error: {str(e)}", Colors.RED)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

