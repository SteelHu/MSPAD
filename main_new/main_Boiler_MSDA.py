"""
多尺度域自适应DACAD - Boiler数据集实验主入口
============================================
功能：在Boiler数据集上测试多尺度域自适应DACAD模型

使用方法：
python main_new/main_Boiler_MSDA.py
"""

import os
import subprocess

if __name__ == '__main__':
    # ============ 第一步：加载Boiler数据集的文件列表 ============
    all_files = os.listdir(os.path.join('datasets', 'Boiler'))
    
    # 提取所有 .csv 文件的文件名（去掉扩展名）
    files = [name[:-4] for name in all_files if name.endswith('.csv')]
    files = sorted(files)
    
    # ============ 第二步：配置跨域实验 ============
    # 遍历所有源域和目标域的组合
    # 注意：files列表中的文件名已经去掉了.csv扩展名
    for src in ['1_train']:  # 源域：训练集
        for trg in ['1_test']:  # 目标域：测试集
            # 确保源域和目标域不是同一个
            if src != trg:
                print('src: ', src, ' / target: ', trg)
                
                # ============ 第三步：启动训练流程 ============
                train = os.path.join('main_new', 'train.py')
                
                # 训练命令参数（使用newmodel算法）
                # Boiler数据集配置：使用较小的TCN通道数（128-128-128）和MLP维度（256）
                command = [
                    'python', train,
                    '--algo_name', 'newmodel',           # 使用新模型
                    '--num_epochs', '20',                # 训练轮数：20 epochs
                    '--queue_size', '98304',             # MoCo队列大小：98304个负样本
                    '--momentum', '0.99',                # 动量编码器更新系数：0.99
                    '--batch_size', '256',               # 训练批次大小：256
                    '--eval_batch_size', '256',          # 验证批次大小：256
                    '--learning_rate', '1e-4',           # 学习率：0.0001
                    '--dropout', '0.2',                  # Dropout率：0.2（Boiler数据集使用）
                    '--weight_decay', '1e-4',            # 权重衰减（L2正则化）：0.0001
                    '--num_channels_TCN', '128-128-128', # TCN通道数：Boiler数据集使用较小配置
                    '--dilation_factor_TCN', '3',        # TCN膨胀因子：3
                    '--kernel_size_TCN', '7',            # TCN卷积核大小：7
                    '--hidden_dim_MLP', '256',           # MLP隐藏层维度：256（Boiler数据集使用）
                    '--weight_loss_disc', '0.5',         # 单尺度域对抗损失权重
                    '--weight_loss_ms_disc', '0.3',      # 多尺度域对抗损失权重（新增）
                    '--weight_loss_pred', '1.0',         # 原型网络分类损失权重（替换Deep SVDD）
                    '--weight_loss_src_sup', '0.1',      # 源域监督对比损失权重
                    '--weight_loss_trg_inj', '0.1',      # 目标域注入对比损失权重
                    '--path_src', 'datasets/Boiler',     # 源域数据路径
                    '--path_trg', 'datasets/Boiler',     # 目标域数据路径
                    '--id_src', src,                     # 源域ID
                    '--id_trg', trg,                     # 目标域ID
                    '--experiment_folder', 'Boiler_MSDA' # 实验结果保存文件夹（多尺度域自适应）
                ]
                
                # 执行训练
                subprocess.run(command)
                
                # ============ 第四步：启动评估流程 ============
                test = os.path.join('main_new', 'eval.py')
                
                # 评估命令参数
                command1 = [
                    'python', test,
                    '--experiments_main_folder', 'results',  # 结果根目录
                    '--experiment_folder', 'Boiler_MSDA',    # 实验子目录
                    '--id_src', src,                         # 源域ID
                    '--id_trg', trg                          # 目标域ID
                ]
                
                # 执行评估
                subprocess.run(command1)

