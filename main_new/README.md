# MSPAD: Multi-Scale Domain Adversarial Prototypical Anomaly Detection

## 概述

MSPAD是DACAD的改进版本，实现了**多尺度域对抗训练**和**原型网络分类器**。

### 核心改进

1. **多尺度域判别器**: 在TCN的每个block后添加域判别器，实现层次化域对齐
2. **加权多尺度损失**: 不同层使用不同权重（低层权重小，高层权重大）
3. **保留原有功能**: 兼容原始DACAD的所有功能

## 文件结构

```
main_new/
├── models/
│   └── MSPAD.py          # MSPAD模型实现
├── algorithms.py             # 算法封装（MSPAD类）
├── train.py                 # 训练脚本
├── eval.py                  # 评估脚本
├── main_MSL_MSPAD.py        # MSL数据集实验入口
└── README.md               # 本文件
```

## 使用方法

### 1. 训练模型

#### 方法一：使用示例脚本（推荐）

```bash
python main_new/main_MSL_MSPAD.py
```

#### 方法二：直接使用train.py

```bash
python main_new/train.py \
    --algo_name MSPAD \
    --num_epochs 20 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --num_channels_TCN 128-256-512 \
    --weight_loss_ms_disc 0.3 \
    --id_src F-5 \
    --id_trg C-1 \
    --experiment_folder MSL_MSDA
```

### 2. 评估模型

```bash
python main_new/eval.py \
    --experiments_main_folder results \
    --experiment_folder MSL_MSDA \
    --id_src F-5 \
    --id_trg C-1
```

## 关键参数说明

### 新增参数

- `--algo_name MSPAD`: 使用MSPAD模型（必须）
- `--weight_loss_ms_disc 0.3`: 多尺度域对抗损失权重（默认0.3）

### 原有参数（保持不变）

- `--weight_loss_disc 0.5`: 单尺度域对抗损失权重
- `--weight_loss_pred 1.0`: 原型网络分类损失权重（替换Deep SVDD）
- `--weight_loss_src_sup 0.1`: 源域监督对比损失权重
- `--weight_loss_trg_inj 0.1`: 目标域注入对比损失权重
- `--prototypical_margin 1.0`: 原型网络间隔参数，控制正常和异常样本之间的最小距离（新增）

## 模型架构

```
输入时间序列
    │
    ├─> TCN Block 1 (128 channels)
    │     └─> 域判别器1 (权重: 0.1)
    │
    ├─> TCN Block 2 (256 channels)
    │     └─> 域判别器2 (权重: 0.3)
    │
    └─> TCN Block 3 (512 channels)
          └─> 域判别器3 (权重: 0.6)
          └─> 单尺度域判别器 (权重: 0.5)
```

## 损失函数

总损失 = 
- 0.5 × 单尺度域对抗损失
- + 0.3 × 多尺度域对抗损失（新增）
- + 1.0 × 原型网络分类损失（替换Deep SVDD）
- + 0.1 × 源域监督对比损失
- + 0.1 × 目标域注入对比损失

## 预期效果

- **域对齐能力提升**: 多尺度对齐减少不同层次的域差异
- **特征表示更鲁棒**: 低层特征对齐提升泛化能力
- **性能提升**: 预期AUPRC提升2-5%

## 注意事项

1. 确保数据集路径正确（默认：`datasets/MSL_SMAP`）
2. 训练结果保存在 `results/MSL_MSDA/{src}-{trg}/` 目录
3. 多尺度权重会根据TCN层数自动调整

## 与原始DACAD的区别

| 特性 | 原始DACAD | MSPAD |
|------|-----------|-------|
| 域判别器数量 | 1个（最终层） | N+1个（N个中间层 + 1个最终层） |
| 域对齐方式 | 单尺度 | 多尺度层次化对齐 |
| 分类器 | Deep SVDD | 原型网络（Prototypical Network） |
| 损失函数 | 4项 | 5项（新增多尺度域对抗损失） |
| 模型文件 | `models/dacad.py` | `models/MSPAD.py` |
| 算法类 | `DACAD` | `MSPAD` |

## 故障排除

### 导入错误

如果遇到导入错误，确保：
1. 项目根目录在Python路径中
2. `main/models/dacad.py` 文件存在（用于导入基础组件）

### 内存不足

如果遇到内存不足：
1. 减小 `--batch_size`（如256 → 128）
2. 减小 `--queue_size`（如98304 → 49152）
3. 减少TCN层数（如 `128-256-512` → `128-256`）

## 引用

如果使用本改进方案，请同时引用原始DACAD论文：

```bibtex
@article{darban2025dacad,
  author={Darban, Zahra Zamanzadeh and Yang, Yiyuan and Webb, Geoffrey I. and Aggarwal, Charu C. and Wen, Qingsong and Salehi, Mahsa},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={DACAD: Domain Adaptation Contrastive Learning for Anomaly Detection in Multivariate Time Series}, 
  year={2025}
}
```

