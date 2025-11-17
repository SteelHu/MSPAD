"""
SMD数据集CLUDA实验主入口
====================
功能：在SMD (Server Machine Dataset) 数据集上使用CLUDA算法进行跨域异常检测实验

实验设置：
- 源域 (Source Domain): 有标签的机器数据，用于训练
- 目标域 (Target Domain): 无标签或少量标签的机器数据，用于测试迁移能力
"""

import os
import subprocess

if __name__ == '__main__':
    # ============ 第一步：加载SMD数据集的文件列表 ============
    all_files = os.listdir(os.path.join('datasets', 'SMD/test'))
    files = [file for file in all_files if file.startswith('machine-')]
    files = sorted(files)
    
    # ============ 第二步：配置跨域实验 ============
    # 源域列表：这些是有充分标注数据的机器
    for src in ['machine-1-1.txt', 'machine-2-3.txt', 'machine-3-7.txt', 'machine-1-5.txt']:
        src = src.replace('machine-', '')
        src = src.replace('.txt', '')
        
        # 目标域列表：这些是要迁移到的机器（标注少或无标注）
        for trg in files:
            trg = trg.replace('machine-', '')
            trg = trg.replace('.txt', '')
            
            if src != trg:
                print('src: ', src, ' / target: ', trg)
                
                # ============ 第三步：启动训练流程 ============
                train = os.path.join('main_cluda', 'train.py')
                
                # CLUDA训练命令参数（SMD数据集）
                command = [
                    'python', train,
                    '--algo_name', 'cluda',              # 算法名称：CLUDA
                    '--num_epochs', '20',                 # 训练轮数：20 epochs
                    '--num_steps', '1000',                # 每个epoch的最大步数
                    '--checkpoint_freq', '100',           # 验证检查点频率
                    '--queue_size', '98304',              # MoCo队列大小：98304个负样本
                    '--momentum', '0.99',                 # 动量编码器更新系数：0.99
                    '--batch_size', '128',                # 训练批次大小：128（SMD数据集较小）
                    '--eval_batch_size', '256',           # 验证批次大小：256
                    '--learning_rate', '1e-4',           # 学习率：0.0001
                    '--dropout', '0.1',                   # Dropout率：0.1
                    '--weight_decay', '1e-2',             # 权重衰减（L2正则化）：0.01
                    '--num_channels_TCN', '128-256-512',  # TCN通道数：逐层递增
                    '--dilation_factor_TCN', '3',         # TCN膨胀因子：3
                    '--kernel_size_TCN', '7',             # TCN卷积核大小：7
                    '--hidden_dim_MLP', '1024',           # MLP隐藏层维度：1024
                    '--weight_loss_src', '1.0',          # 源域对比损失权重
                    '--weight_loss_trg', '1.0',          # 目标域对比损失权重
                    '--weight_loss_ts', '1.0',           # 跨域对比损失权重
                    '--weight_loss_disc', '1.0',         # 域判别损失权重
                    '--weight_loss_pred', '1.0',         # 预测损失权重
                    '--use_mask',                         # 使用mask（CLUDA需要）
                    '--path_src', 'datasets/SMD/test',     # 源域数据路径
                    '--path_trg', 'datasets/SMD/test',     # 目标域数据路径
                    '--id_src', src,                     # 源域机器ID
                    '--id_trg', trg,                     # 目标域机器ID
                    '--experiments_main_folder', 'results',  # 结果根目录
                    '--experiment_folder', 'SMD_cluda'   # 实验结果保存文件夹（区分CLUDA）
                ]
                
                # 执行训练（会保存模型到 results/SMD_cluda/{src}-{trg}/ 目录）
                subprocess.run(command)

                # ============ 第四步：启动评估流程 ============
                test = os.path.join('main_cluda', 'eval.py')
                
                # 评估命令参数
                command1 = [
                    'python', test,
                    '--experiments_main_folder', 'results',  # 结果根目录
                    '--experiment_folder', 'SMD_cluda',      # 实验子目录
                    '--id_src', src,                         # 源域ID
                    '--id_trg', trg                          # 目标域ID
                ]
                
                # 执行评估（加载训练好的模型，在测试集上评估性能）
                subprocess.run(command1)

