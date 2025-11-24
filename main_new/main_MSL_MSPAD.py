"""
MSPAD - MSL数据集实验主入口
===========================
功能：在MSL数据集上测试MSPAD模型
MSPAD: Multi-Scale Domain Adversarial Prototypical Anomaly Detection

使用方法：
python main_new/main_MSL_MSPAD.py
"""

import os
import subprocess
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # ============ 第一步：加载MSL数据集的文件列表 ============
    all_files = os.listdir(os.path.join('datasets', 'MSL_SMAP/test'))
    all_names = [name[:-4] for name in all_files if name.endswith('.npy')]
    
    # 读取标注的异常信息 CSV 文件
    with open(os.path.join('datasets/MSL_SMAP/', 'labeled_anomalies.csv'), 'r') as file:
        csv_reader = pd.read_csv(file, delimiter=',')
    
    # 筛选出 MSL 航天器的数据
    data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
    space_files = np.asarray(data_info['chan_id'])
    
    # 找出既有数据文件又有标注的通道
    files = [file for file in all_names if file in space_files]
    files = sorted(files)
    
    # ============ 第二步：配置跨域实验 ============
    # 源域列表：这些是有充分标注数据的通道
    for src in ['F-5']:
        # 目标域列表：这些是要迁移到的通道（标注少或无标注）
        for trg in ['C-1']:
            if src != trg:
                print('src: ', src, ' / target: ', trg)
                
                # ============ 第三步：启动训练流程 ============
                train = os.path.join('main_new', 'train.py')
                
                # 训练命令参数（使用MSPAD模型）
                command = [
                    'python', train,
                    '--algo_name', 'MSPAD',           # 使用MSPAD模型
                    '--num_epochs', '20',                # 训练轮数：20 epochs
                    '--queue_size', '98304',             # MoCo队列大小：98304个负样本
                    '--momentum', '0.99',                # 动量编码器更新系数：0.99
                    '--batch_size', '256',               # 训练批次大小：256
                    '--eval_batch_size', '256',          # 验证批次大小：256
                    '--learning_rate', '1e-4',           # 学习率：0.0001
                    '--dropout', '0.1',                  # Dropout率：0.1
                    '--weight_decay', '1e-4',            # 权重衰减（L2正则化）：0.0001
                    '--num_channels_TCN', '128-256-512', # TCN通道数：逐层递增
                    '--dilation_factor_TCN', '3',        # TCN膨胀因子：3
                    '--kernel_size_TCN', '7',            # TCN卷积核大小：7
                    '--hidden_dim_MLP', '1024',          # MLP隐藏层维度：1024
                    '--weight_loss_disc', '0.0',         # 单尺度域对抗损失权重（已弃用）
                    '--weight_loss_ms_disc', '0.3',      # 多尺度域对抗损失权重（新增）
                    '--weight_loss_pred', '1.0',         # 原型网络分类损失权重（替换Deep SVDD）
                    '--weight_loss_src_sup', '0.1',      # 源域监督对比损失权重
                    '--weight_loss_trg_inj', '0.1',      # 目标域注入对比损失权重
                    '--id_src', src,                     # 源域通道ID
                    '--id_trg', trg,                     # 目标域通道ID
                    '--path_src', 'datasets/MSL_SMAP',   # 源域数据路径
                    '--path_trg', 'datasets/MSL_SMAP',   # 目标域数据路径
                    '--experiment_folder', 'MSL_MSDA'    # 实验结果保存文件夹（MSPAD）
                ]
                
                # 执行训练
                subprocess.run(command)
                
                # ============ 第四步：启动评估流程 ============
                test = os.path.join('main_new', 'eval.py')
                
                # 评估命令参数
                command1 = [
                    'python', test,
                    '--experiments_main_folder', 'results',  # 结果根目录
                    '--experiment_folder', 'MSL_MSDA',       # 实验子目录
                    '--id_src', src,                          # 源域ID
                    '--id_trg', trg                           # 目标域ID
                ]
                
                # 执行评估
                subprocess.run(command1)

