"""
DACAD 训练主流程
=================
功能：执行双域（源域+目标域）联合训练，实现跨域异常检测

核心流程：
1. 加载源域和目标域的训练/验证数据
2. 初始化 DACAD 模型（包括编码器、分类器、判别器）
3. 双域联合训练：
   - 源域：有监督学习（使用标签）
   - 目标域：自监督学习（使用异常注入）
4. 每个 epoch 结束后在验证集上评估性能
5. 保存最佳模型

关键技术：
- 对比学习 (Contrastive Learning)
- 域对抗训练 (Domain Adversarial Training)  
- 异常注入 (Anomaly Injection)
- Deep SVDD 分类器
"""

import sys
import os
# 添加项目根目录到 Python 路径
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
    # 设置 PyTorch 随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 设置 NumPy 和 Python 随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 注：可以设置确定性模式，但会降低性能
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # ============ 第二步：配置日志记录器 ============
    # 日志文件保存路径：results/{experiment_folder}/{id_src}-{id_trg}/train.log
    log = get_logger(
        os.path.join(args.experiments_main_folder, args.experiment_folder, str(args.id_src) + "-" + str(args.id_trg),
                     args.log))

    # 获取数据集类型（smd/msl/boiler），用于选择合适的评估指标
    dataset_type = get_dataset_type(args)

    def log_scores(args, dataset_type, metrics_pred):
        """
        记录评估指标到日志
        
        不同数据集使用不同的评估指标：
        - SMD/MSL/Boiler: AUPRC (精确率-召回率曲线下面积) 和 F1
        - 其他: 准确率、宏平均F1、加权F1
        """
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
    batch_size = args.batch_size                    # 训练批次大小
    eval_batch_size = args.eval_batch_size          # 验证批次大小
    num_val_iteration = args.num_val_iteration      # 验证时使用的批次数量

    # ============ 第四步：加载源域和目标域的数据集 ============
    # 源域数据（有标签）
    dataset_src = get_dataset(args, domain_type="source", split_type="train")      # 源域训练集
    dataset_val_src = get_dataset(args, domain_type="source", split_type="val")    # 源域验证集

    # 目标域数据（无标签或少量标签）
    dataset_trg = get_dataset(args, domain_type="target", split_type="train")      # 目标域训练集
    dataset_val_trg = get_dataset(args, domain_type="target", split_type="val")    # 目标域验证集

    # 创建数据加载器
    # drop_last=True: 丢弃最后不完整的批次，因为 MoCo 队列需要固定的批次大小
    dataloader_src = DataLoader(dataset_src, batch_size=batch_size,
                                shuffle=True, num_workers=0, drop_last=True)
    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)
    dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)

    # 计算验证时使用的迭代次数（取两个域的较小值）
    max_num_val_iteration = min(len(dataloader_val_src), len(dataloader_val_trg))
    if max_num_val_iteration < num_val_iteration:
        num_val_iteration = max_num_val_iteration

    # ============ 第五步：获取数据维度信息 ============
    # 输入通道数（时间序列的特征维度）
    input_channels_dim = dataset_src[0]['sequence'].shape[1]
    # 静态特征维度（如果有的话）
    input_static_dim = dataset_src[0]['static'].shape[0] if 'static' in dataset_src[0] else 0

    # ============ 第六步：初始化 DACAD 算法 ============
    # 创建模型，包括编码器、分类器、判别器等所有组件
    algorithm = get_algorithm(args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim)

    # 实验结果保存路径
    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(args.id_trg))

    # ============ 第七步：初始化训练状态 ============
    count_step = 0           # 训练步数计数器
    best_val_score = -100    # 记录最佳验证性能（用于保存最佳模型）

    # 获取数据的统计信息（均值和标准差），用于归一化
    src_mean, src_std = dataset_src.get_statistic()
    trg_mean, trg_std = dataset_src.get_statistic()
    
    # ============ 第八步：开始训练循环 ============
    for i in range(args.num_epochs):
        # 每个 epoch 重新创建目标域数据加载器（重新打乱数据）
        dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                    shuffle=True, num_workers=0, drop_last=True)
        dataloader_iterator = iter(dataloader_trg)

        # 遍历源域数据
        for i_batch, sample_batched_src in enumerate(dataloader_src):
                sample_batched_src = sample_batched_src
                # 可选：将数据移到 GPU（当前已在 dataset 中处理）
                for key, value in sample_batched_src.items():
                    sample_batched_src[key] = sample_batched_src[key]
                
                # 确保批次大小一致（MoCo 队列指针需要固定批次大小）
                if len(sample_batched_src['sequence']) != batch_size:
                    continue

                # 获取对应的目标域数据批次
                try:
                    sample_batched_trg = next(dataloader_iterator)
                except StopIteration:
                    # 目标域数据用完时，重新创建迭代器
                    dataloader_trg = DataLoader(dataset_trg, batch_size=batch_size,
                                                shuffle=True, num_workers=0, drop_last=True)
                    dataloader_iterator = iter(dataloader_trg)
                    sample_batched_trg = next(dataloader_iterator)

                for key, value in sample_batched_trg.items():
                    sample_batched_trg[key] = sample_batched_trg[key]

                # 确保目标域批次大小也一致
                if len(sample_batched_trg['sequence']) != batch_size:
                    continue

                # ========== 核心：执行一步训练 ==========
                # 同时处理源域和目标域数据
                # - 源域：监督学习 + 域对抗
                # - 目标域：自监督对比学习 + 域对抗
                algorithm.step(sample_batched_src, sample_batched_trg, count_step=count_step, epoch=i,
                               src_mean=src_mean, src_std=src_std, trg_mean=trg_mean, trg_std=trg_std)

                count_step += 1
                # ========== 每个 epoch 结束时验证模型 ==========
                if count_step % len(dataloader_src) == 0:
                    # 显示训练进度和损失
                    progress = ProgressMeter(
                        len(dataloader_src),
                        algorithm.return_metrics(),
                        prefix="Epoch: [{}]".format(i))
                    log(progress.display(i_batch + 1, is_logged=True))

                    # 重置训练指标记录器
                    algorithm.init_metrics()
                    algorithm.init_pred_meters_val()

                    # ========== 切换到评估模式 ==========
                    algorithm.eval()

                    # 创建验证集数据加载器
                    dataloader_val_src = DataLoader(dataset_val_src, batch_size=eval_batch_size,
                                                    shuffle=True, num_workers=0, drop_last=True)
                    dataloader_val_src_iterator = iter(dataloader_val_src)

                    dataloader_val_trg = DataLoader(dataset_val_trg, batch_size=eval_batch_size,
                                                    shuffle=True, num_workers=0, drop_last=True)
                    dataloader_val_trg_iterator = iter(dataloader_val_trg)

                    # 在验证集上运行模型（不更新参数）
                    for i_batch_val in range(num_val_iteration):
                        sample_batched_val_src = next(dataloader_val_src_iterator)
                        sample_batched_val_trg = next(dataloader_val_trg_iterator)

                        # 验证步骤（forward only，no backward）
                        algorithm.step(sample_batched_val_src, sample_batched_val_trg, count_step=count_step,
                                       src_mean=src_mean, src_std=src_std, trg_mean=trg_mean, trg_std=trg_std)

                    # 显示验证进度
                    progress_val = ProgressMeter(
                        num_val_iteration,
                        algorithm.return_metrics(),
                        prefix="Epoch: [{}]".format(i))

                    # 获取验证集上的预测指标
                    metrics_pred_val_src = algorithm.pred_meter_val_src.get_metrics()
                    metrics_pred_val_trg = algorithm.pred_meter_val_trg.get_metrics()

                    # 记录验证结果
                    log("VALIDATION RESULTS")
                    log(progress_val.display(i_batch_val + 1, is_logged=True))

                    log("VALIDATION SOURCE PREDICTIONS")
                    log_scores(args, dataset_type, metrics_pred_val_src)

                    # ========== 根据数据集类型选择主要评估指标 ==========
                    if dataset_type == "msl":
                        cur_val_score = metrics_pred_val_src["best_f1"]      # MSL: 使用源域 F1
                    elif dataset_type == "boiler":
                        cur_val_score = metrics_pred_val_src["best_f1"]     # Boiler: 使用源域 F1
                    elif dataset_type == "smd":
                        cur_val_score = metrics_pred_val_trg["best_f1"]     # SMD: 使用目标域 F1
                    else:
                        cur_val_score = metrics_pred_val_src["mac_f1"]      # 其他: 使用宏平均 F1

                    # ========== 保存最佳模型 ==========
                    if cur_val_score > best_val_score:
                        # 保存模型权重到 experiment_folder_path/model_best.pth.tar
                        algorithm.save_state(experiment_folder_path)
                        best_val_score = cur_val_score
                        log(f"*** New best model saved! Best score: {best_val_score:.4f} ***")
                    
                    log("VALIDATION TARGET PREDICTIONS")
                    log_scores(args, dataset_type, metrics_pred_val_trg)

                    # ========== 切换回训练模式 ==========
                    algorithm.train()

                    # 重置指标记录器，准备下一个 epoch
                    algorithm.init_metrics()

                else:
                    continue
                break  # 每个 epoch 只训练一遍源域数据


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = ArgumentParser(description="parse args")

    parser.add_argument('--algo_name', type=str, default='dacad')

    parser.add_argument('-dr', '--dropout', type=float, default=0.1)
    parser.add_argument('-mo', '--momentum', type=float, default=0.99)  # DACAD
    parser.add_argument('-qs', '--queue_size', type=int, default=98304)  # DACAD
    parser.add_argument('--use_batch_norm', action='store_true')
    parser.add_argument('--use_mask', action='store_true')  # DACAD
    parser.add_argument('-wr', '--weight_ratio', type=float, default=10.0)
    parser.add_argument('-bs', '--batch_size', type=int, default=200)  # 2048)
    parser.add_argument('-ebs', '--eval_batch_size', type=int, default=200)  # 2048)
    parser.add_argument('-nvi', '--num_val_iteration', type=int, default=50)
    parser.add_argument('-ne', '--num_epochs', type=int, default=10)
    parser.add_argument('-ns', '--num_steps', type=int, default=1000)
    parser.add_argument('-cf', '--checkpoint_freq', type=int, default=1000)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-ws', '--warmup_steps', type=int, default=2000)
    parser.add_argument('--num_channels_TCN', type=str, default='64-64-64-64-64')  # All TCN models
    parser.add_argument('--kernel_size_TCN', type=int, default=3)  # All TCN models
    parser.add_argument('--dilation_factor_TCN', type=int, default=2)  # All TCN models
    parser.add_argument('--stride_TCN', type=int, default=1)  # All TCN models
    parser.add_argument('--hidden_dim_MLP', type=int, default=256)  # All classifier and discriminators

    # The weight of the domain classification loss
    parser.add_argument('-w_d', '--weight_domain', type=float, default=0.1)
    # Below weights are defined for DACAD
    parser.add_argument('--weight_loss_src', type=float, default=0.0)
    parser.add_argument('--weight_loss_trg', type=float, default=0.0)
    parser.add_argument('--weight_loss_ts', type=float, default=0.0)
    parser.add_argument('--weight_loss_disc', type=float, default=0.5)
    parser.add_argument('--weight_loss_pred', type=float, default=1.0)
    parser.add_argument('--weight_loss_src_sup', type=float, default=0.1)
    parser.add_argument('--weight_loss_trg_inj', type=float, default=0.1)

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='smd')

    parser.add_argument('--path_src', type=str, default='datasets/MSL_SMAP') #../datasets/Boiler/   ../datasets/MSL_SMAP
    parser.add_argument('--path_trg', type=str, default='datasets/MSL_SMAP') #../datasets/SMD/test
    parser.add_argument('--age_src', type=int, default=-1)
    parser.add_argument('--age_trg', type=int, default=-1)
    parser.add_argument('--id_src', type=str, default='1-5')
    parser.add_argument('--id_trg', type=str, default='1-1')

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