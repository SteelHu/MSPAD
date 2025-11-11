"""
多尺度域自适应DACAD - SMD数据集实验主入口
==========================================
功能：在SMD (Server Machine Dataset) 数据集上测试多尺度域自适应DACAD模型

使用方法：
python main_new/main_SMD_MSDA.py

注意：需要根据实际路径修改数据集路径
"""

import os
import subprocess

if __name__ == '__main__':
    # ============ 第一步：加载SMD数据集的文件列表 ============
    # 注意：根据实际数据集路径调整
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join('datasets', 'SMD', 'test'),
        os.path.join('..', '..', 'datasets', 'SMD', 'test'),
        os.path.join('..', 'datasets', 'SMD', 'test'),
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("Error: SMD dataset path not found!")
        print("Please check if the dataset exists in one of these locations:")
        for path in possible_paths:
            print(f"  - {os.path.abspath(path)}")
        exit(1)
    
    all_files = os.listdir(dataset_path)
    
    # 提取所有以 'machine-' 开头的文件
    files = [file for file in all_files if file.startswith('machine-')]
    files = sorted(files)
    
    # ============ 第二步：配置跨域实验 ============
    # 源域列表：这些是有充分标注数据的服务器
    # 原始配置使用：['machine-1-1.txt', 'machine-2-3.txt', 'machine-3-7.txt', 'machine-1-5.txt']
    source_machines = ['machine-1-1.txt']
    
    for src_file in source_machines:
        # 提取源域ID（去掉 'machine-' 前缀和 '.txt' 后缀）
        src = src_file.replace('machine-', '').replace('.txt', '')
        
        # 遍历所有目标域
        for trg_file in ['machine-2-3.txt']:
            # 提取目标域ID
            trg = trg_file.replace('machine-', '').replace('.txt', '')
            
            # 确保源域和目标域不是同一个
            if src != trg:
                print('src: ', src, ' / target: ', trg)
                
                # ============ 第三步：启动训练流程 ============
                train = os.path.join('main_new', 'train.py')
                
                # 训练命令参数（使用newmodel算法）
                # SMD数据集配置：使用较大的TCN通道数（128-256-512）和MLP维度（1024）
                command = [
                    'python', train,
                    '--algo_name', 'newmodel',           # 使用新模型
                    '--num_epochs', '20',                # 训练轮数：20 epochs
                    '--queue_size', '98304',             # MoCo队列大小：98304个负样本
                    '--momentum', '0.99',                # 动量编码器更新系数：0.99
                    '--batch_size', '128',               # 训练批次大小：128（SMD数据集使用较小batch）
                    '--eval_batch_size', '256',          # 验证批次大小：256
                    '--learning_rate', '1e-4',           # 学习率：0.0001
                    '--dropout', '0.1',                  # Dropout率：0.1
                    '--weight_decay', '1e-4',            # 权重衰减（L2正则化）：0.0001
                    '--num_channels_TCN', '128-256-512', # TCN通道数：逐层递增（SMD数据集使用）
                    '--dilation_factor_TCN', '3',        # TCN膨胀因子：3
                    '--kernel_size_TCN', '7',            # TCN卷积核大小：7
                    '--hidden_dim_MLP', '1024',          # MLP隐藏层维度：1024（SMD数据集使用）
                    '--weight_loss_disc', '0.5',         # 单尺度域对抗损失权重
                    '--weight_loss_ms_disc', '0.3',      # 多尺度域对抗损失权重（新增）
                    '--weight_loss_pred', '1.0',         # 原型网络分类损失权重（替换Deep SVDD）
                    '--weight_loss_src_sup', '0.1',      # 源域监督对比损失权重
                    '--weight_loss_trg_inj', '0.1',      # 目标域注入对比损失权重
                    '--id_src', src,                     # 源域ID
                    '--id_trg', trg,                     # 目标域ID
                    '--path_src', 'datasets/SMD/test',        # 源域数据路径
                    '--path_trg', 'datasets/SMD/test',        # 目标域数据路径
                    '--experiment_folder', 'SMD_MSDA'    # 实验结果保存文件夹（多尺度域自适应）
                ]
                
                # 注意：如果数据集路径不同，需要调整path_src和path_trg
                # 例如，如果数据集在上级目录：
                # '--path_src', '../../datasets/SMD',
                # '--path_trg', '../../datasets/SMD',
                
                # 执行训练
                subprocess.run(command)
                
                # ============ 第四步：启动评估流程 ============
                test = os.path.join('main_new', 'eval.py')
                
                # 评估命令参数
                command1 = [
                    'python', test,
                    '--experiments_main_folder', 'results',  # 结果根目录
                    '--experiment_folder', 'SMD_MSDA',       # 实验子目录
                    '--id_src', src,                         # 源域ID
                    '--id_trg', trg                          # 目标域ID
                ]
                
                # 执行评估
                subprocess.run(command1)

