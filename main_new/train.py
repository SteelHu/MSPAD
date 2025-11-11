"""
多尺度域自适应DACAD训练主流程
==============================
功能：执行多尺度域对抗训练，实现跨域异常检测

核心改进：
- 在TCN的多个中间层同时进行域对抗训练
- 层次化域对齐：从低层到高层逐步对齐域特征
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import random
import torch
from utils.dataset import get_dataset
from torch.utils.data import DataLoader
from utils.util_progress_log import ProgressMeter, get_logger, get_dataset_type
import json
from argparse import ArgumentParser
from algorithms import get_algorithm


def main(args):
    """
    训练主函数
    
    参数:
        args: 命令行参数，包含所有超参数配置
    """
    # ============ 第一步：设置随机种子，确保实验可复现 ============
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # ============ 第二步：配置日志记录器 ============
    log = get_logger(
        os.path.join(args.experiments_main_folder, args.experiment_folder, 
                     str(args.id_src) + "-" + str(args.id_trg), args.log))
    
    dataset_type = get_dataset_type(args)
    
    def log_scores(args, dataset_type, metrics_pred):
        """记录评估指标到日志"""
        if dataset_type == "smd":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        elif dataset_type == "msl":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        elif dataset_type == "boiler":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
        else:
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))
    
    # ============ 第三步：设置训练参数 ============
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    num_val_iteration = args.num_val_iteration
    
    # ============ 第四步：加载源域和目标域的数据集 ============
    dataset_src = get_dataset(args, domain_type="source", split_type="train")
    dataset_val_src = get_dataset(args, domain_type="source", split_type="val")
    
    dataset_trg = get_dataset(args, domain_type="target", split_type="train")
    dataset_val_trg = get_dataset(args, domain_type="target", split_type="val")
    
    # 创建数据加载器
    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                                shuffle=True, num_workers=0, drop_last=True)
    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)
    dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)
    
    max_num_val_iteration = min(len(dataloader_val_src), len(dataloader_val_trg))
    if max_num_val_iteration < num_val_iteration:
        num_val_iteration = max_num_val_iteration
    
    # ============ 第五步：获取数据维度信息 ============
    input_channels_dim = dataset_src[0]['sequence'].shape[1]
    input_static_dim = dataset_src[0]['static'].shape[0] if 'static' in dataset_src[0] else 0
    
    # ============ 第六步：初始化多尺度域自适应DACAD算法 ============
    algorithm = get_algorithm(args, input_channels_dim=input_channels_dim, 
                             input_static_dim=input_static_dim)
    
    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(args.id_trg))
    
    # ============ 第七步：初始化训练状态 ============
    count_step = 0
    best_val_score = -100
    
    src_mean, src_std = dataset_src.get_statistic()
    trg_mean, trg_std = dataset_trg.get_statistic()
    
    # ============ 第八步：开始训练循环 ============
    for i in range(args.num_epochs):
        dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)
        dataloader_iterator = iter(dataloader_trg)
        
        for i_batch, sample_batched_src in enumerate(dataloader_src):
            sample_batched_src = sample_batched_src
            for key, value in sample_batched_src.items():
                sample_batched_src[key] = sample_batched_src[key]
            
            if len(sample_batched_src['sequence']) != batch_size:
                continue
            
            try:
                sample_batched_trg = next(dataloader_iterator)
            except StopIteration:
                dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                            shuffle=True, num_workers=0, drop_last=True)
                dataloader_iterator = iter(dataloader_trg)
                sample_batched_trg = next(dataloader_iterator)
            
            for key, value in sample_batched_trg.items():
                sample_batched_trg[key] = sample_batched_trg[key]
            
            if len(sample_batched_trg['sequence']) != batch_size:
                continue
            
            # ========== 核心：执行一步训练 ==========
            algorithm.step(sample_batched_src, sample_batched_trg, count_step=count_step, 
                          epoch=i, src_mean=src_mean, src_std=src_std, 
                          trg_mean=trg_mean, trg_std=trg_std)
            
            count_step += 1
            
            # ========== 每个epoch结束时验证模型 ==========
            if count_step % len(dataloader_src) == 0:
                progress = ProgressMeter(
                    len(dataloader_src),
                    algorithm.return_metrics(),
                    prefix="Epoch: [{}]".format(i))
                log(progress.display(i_batch + 1, is_logged=True))
                
                algorithm.init_metrics()
                algorithm.init_pred_meters_val()
                
                algorithm.eval()
                
                dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                                shuffle=True, num_workers=0, drop_last=True)
                dataloader_val_src_iterator = iter(dataloader_val_src)
                
                dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                                shuffle=True, num_workers=0, drop_last=True)
                dataloader_val_trg_iterator = iter(dataloader_val_trg)
                
                for i_batch_val in range(num_val_iteration):
                    sample_batched_val_src = next(dataloader_val_src_iterator)
                    sample_batched_val_trg = next(dataloader_val_trg_iterator)
                    
                    algorithm.step(sample_batched_val_src, sample_batched_val_trg, 
                                  count_step=count_step, src_mean=src_mean, src_std=src_std,
                                  trg_mean=trg_mean, trg_std=trg_std)
                
                progress_val = ProgressMeter(
                    num_val_iteration,
                    algorithm.return_metrics(),
                    prefix="Epoch: [{}]".format(i))
                
                metrics_pred_val_src = algorithm.pred_meter_val_src.get_metrics()
                metrics_pred_val_trg = algorithm.pred_meter_val_trg.get_metrics()
                
                log("VALIDATION RESULTS")
                log(progress_val.display(i_batch_val + 1, is_logged=True))
                
                log("VALIDATION SOURCE PREDICTIONS")
                log_scores(args, dataset_type, metrics_pred_val_src)
                
                if dataset_type == "msl":
                    cur_val_score = metrics_pred_val_src["best_f1"]
                elif dataset_type == "boiler":
                    cur_val_score = metrics_pred_val_src["best_f1"]
                elif dataset_type == "smd":
                    cur_val_score = metrics_pred_val_trg["best_f1"]
                else:
                    cur_val_score = metrics_pred_val_src["mac_f1"]
                
                if cur_val_score > best_val_score:
                    algorithm.save_state(experiment_folder_path)
                    best_val_score = cur_val_score
                    log(f"*** New best model saved! Best score: {best_val_score:.4f} ***")
                
                log("VALIDATION TARGET PREDICTIONS")
                log_scores(args, dataset_type, metrics_pred_val_trg)
                
                algorithm.train()
                algorithm.init_metrics()
            else:
                continue
            break


if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")
    
    parser.add_argument('--algo_name', type=str, default='newmodel')  # 使用新模型
    
    parser.add_argument('-dr', '--dropout', type=float, default=0.1)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99)
    parser.add_argument('-qs', '--queue_size', type=int, default=98304)
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true')
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=200)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--num_channels_TCN', type=str, default='128-256-512')
    parser.add_argument('--kernel_size_TCN', type=int, default=7)
    parser.add_argument('--dilation_factor_TCN', type=int, default=3)
    parser.add_argument('--stride_TCN', type=int, default=1)
    parser.add_argument('--hidden_dim_MLP', type=int, default=1024)
    
    # 损失权重
    parser.add_argument('-w_d', '--weight_domain', type=float, default=0.1)
    parser.add_argument('--weight_loss_src', type=float, default=0.0)
    parser.add_argument('--weight_loss_trg', type=float, default=0.0)
    parser.add_argument('--weight_loss_ts', type=float, default=0.0)
    parser.add_argument('--weight_loss_disc', type=float, default=0.5)
    parser.add_argument('--weight_loss_ms_disc', type=float, default=0.3)  # 多尺度域对抗损失权重（新增）
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)
    parser.add_argument('--weight_loss_src_sup', type=float, default=0.1)
    parser.add_argument('--weight_loss_trg_inj', type=float, default=0.1)
    
    # 原型网络超参数（替换Deep SVDD）
    parser.add_argument('--prototypical_margin', type=float, default=1.0,
                        help='原型网络间隔参数，控制正常和异常样本之间的最小距离（默认1.0）')
    
    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='MSL_MSDA')
    
    parser.add_argument('--path_src', type=str, default='datasets/MSL_SMAP')
    parser.add_argument('--path_trg', type=str, default='datasets/MSL_SMAP')
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=str, default='F-5')
    parser.add_argument('--id_trg', type=str, default='C-1')
    
    parser.add_argument('--task', type=str, default='decompensation')
    parser.add_argument('-l', '--log', type=str, default='train.log')
    parser.add_argument('--seed', type=int, default=1234)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiments_main_folder):
        os.mkdir(args.experiments_main_folder)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder)):
        os.makedirs(os.path.join(args.experiments_main_folder, args.experiment_folder), exist_ok=True)
    if not os.path.exists(os.path.join(args.experiments_main_folder, args.experiment_folder,
                                       str(args.id_src) + "-" + str(args.id_trg))):
        os.mkdir(os.path.join(args.experiments_main_folder, args.experiment_folder,
                              str(args.id_src) + "-" + str(args.id_trg)))
    
    with open(os.path.join(args.experiments_main_folder, args.experiment_folder,
                           str(args.id_src) + "-" + str(args.id_trg), 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    main(args)

