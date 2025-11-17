# 消融实验结果文件夹

## 📋 说明

此文件夹用于保存MSPAD方法的消融实验结果。消融实验验证了多尺度域对抗、原型网络和加权损失等核心组件的有效性。

## 📁 文件命名规则

### 结果CSV文件
- `ablation_results_{dataset}_{src_id}_{trg_id}.csv` - 特定数据集和源-目标对的消融实验结果
  - 例如: `ablation_results_MSL_F-5_C-1.csv`
  - 包含所有配置的性能指标

### 摘要JSON文件
- `ablation_results_{dataset}_{src_id}_{trg_id}.json` - 实验摘要信息
  - 包含实验时间戳、持续时间、配置列表等元数据

## 📊 CSV文件格式

每个CSV文件包含以下列：
- `config_id`: 配置ID（如Abl-4.1）
- `description`: 配置描述
- `best_f1`: 最佳F1分数
- `best_prec`: 最佳精确率
- `best_rec`: 最佳召回率
- `best_thr`: 最佳阈值
- `avg_prc`: 平均精度（AUPRC）
- `roc_auc`: ROC AUC
- `macro_F1`: 宏平均F1分数
- `src_id`: 源域ID
- `trg_id`: 目标域ID

## 🔄 配置说明

CSV中包含25个配置，对应论文表3-6：
- **Abl-4.x**: 核心组件消融（Baseline, Multi-Scale DA, Prototypical Net, Weighted Loss, Full）
- **Abl-5.x**: 多尺度层组合（Layer 1 Only, Layer 2 Only, etc., All Layers）
- **Abl-6.x**: 损失函数消融（Full, w/o Single-Scale DA, w/o Multi-Scale DA, etc.）

## 🚀 使用方法

运行消融实验：
```bash
python experiments/ablation_experiments.py --dataset MSL --src F-5 --trg C-1
```

结果会自动保存到此文件夹，并追加到CSV中。

## 📝 注意事项
- 实验结果基于5次运行的平均值±标准差（论文中报告）
- 建议结合论文表3-6查看详细分析
- 定期备份CSV文件
