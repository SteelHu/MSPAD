"""
Boiler数据集CLUDA实验主入口
====================
功能：在Boiler数据集上使用CLUDA算法进行跨域异常检测实验

实验设置：
- 源域 (Source Domain): 有标签的锅炉数据，用于训练
- 目标域 (Target Domain): 无标签或少量标签的锅炉数据，用于测试迁移能力
"""

import os
import subprocess

if __name__ == '__main__':
    # ============ 第一步：加载Boiler数据集的文件列表 ============
    all_files = os.listdir(os.path.join('datasets', 'Boiler'))
    files = [name[:-4] for name in all_files if name.endswith('.csv')]
    files = sorted(files)
    
    # ============ 第二步：配置跨域实验 ============
    # 遍历所有源域和目标域的组合
    for src in files:
        for trg in files:
            if src != trg:
                print('src: ', src, ' / target: ', trg)

                # ============ 第三步：启动训练流程 ============
                train = os.path.join('main_cluda', 'train.py')
                
                # CLUDA训练命令参数（Boiler数据集）
                command = [
                    'python', train,
                    '--algo_name', 'cluda',              # 算法名称：CLUDA
                    '--num_epochs', '20',                 # 训练轮数：20 epochs
                    '--num_steps', '1000',                # 每个epoch的最大步数
                    '--checkpoint_freq', '100',           # 验证检查点频率
                    '--queue_size', '98304',              # MoCo队列大小：98304个负样本
                    '--momentum', '0.99',                 # 动量编码器更新系数：0.99
                    '--batch_size', '256',                # 训练批次大小：256
                    '--eval_batch_size', '256',           # 验证批次大小：256
                    '--learning_rate', '1e-4',           # 学习率：0.0001
                    '--dropout', '0.2',                   # Dropout率：0.2（Boiler数据集使用0.2）
                    '--weight_decay', '1e-2',             # 权重衰减（L2正则化）：0.01
                    '--num_channels_TCN', '128-128-128',  # TCN通道数：Boiler数据集使用较小通道数
                    '--dilation_factor_TCN', '3',         # TCN膨胀因子：3
                    '--kernel_size_TCN', '7',             # TCN卷积核大小：7
                    '--hidden_dim_MLP', '256',           # MLP隐藏层维度：256（Boiler数据集使用较小维度）
                    '--weight_loss_src', '1.0',          # 源域对比损失权重
                    '--weight_loss_trg', '1.0',          # 目标域对比损失权重
                    '--weight_loss_ts', '1.0',           # 跨域对比损失权重
                    '--weight_loss_disc', '1.0',         # 域判别损失权重
                    '--weight_loss_pred', '1.0',         # 预测损失权重
                    '--use_mask',                         # 使用mask（CLUDA需要）
                    '--path_src', 'datasets/Boiler',     # 源域数据路径
                    '--path_trg', 'datasets/Boiler',     # 目标域数据路径
                    '--id_src', src,                     # 源域文件ID（不含.csv扩展名）
                    '--id_trg', trg,                     # 目标域文件ID（不含.csv扩展名）
                    '--experiments_main_folder', 'results',  # 结果根目录
                    '--experiment_folder', 'Boiler_cluda'   # 实验结果保存文件夹（区分CLUDA）
                ]
                
                # 执行训练（会保存模型到 results/Boiler_cluda/{src}-{trg}/ 目录）
                subprocess.run(command)

                # ============ 第四步：启动评估流程 ============
                test = os.path.join('main_cluda', 'eval.py')
                
                # 评估命令参数
                command1 = [
                    'python', test,
                    '--experiments_main_folder', 'results',  # 结果根目录
                    '--experiment_folder', 'Boiler_cluda',   # 实验子目录
                    '--id_src', src,                         # 源域ID
                    '--id_trg', trg                          # 目标域ID
                ]
                
                # 执行评估（加载训练好的模型，在测试集上评估性能）
                subprocess.run(command1)

