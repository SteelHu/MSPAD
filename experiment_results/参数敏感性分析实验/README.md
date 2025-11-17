# 参数敏感性分析实验结果文件夹

## 📋 说明

此文件夹用于保存MSPAD方法的参数敏感性分析实验结果。

## 📁 文件命名规则

### 汇总结果CSV文件
- `Sensitivity_{dataset}_{src_id}_{trg_id}.csv` - 特定数据集和源-目标对的敏感性分析结果
  - 例如: `Sensitivity_MSL_F-5_C-1.csv`
  - 包含所有参数值的实验结果

### 摘要JSON文件
- `sensitivity_analysis_{dataset}_{src_id}_{trg_id}.json` - 实验摘要信息
  - 包含实验时间戳、持续时间、参数配置等元数据

## 📊 CSV文件格式

每个CSV文件包含以下列：
- `src_id`: 源域ID
- `trg_id`: 目标域ID
- `param_name`: 参数名称（如 `weight_loss_ms_disc`, `prototypical_margin` 等）
- `param_value`: 参数值
- `best_f1`: 最佳F1分数
- `best_prec`: 最佳精确率
- `best_rec`: 最佳召回率
- `best_thr`: 最佳阈值
- `avg_prc`: 平均精度（AUPRC）
- `roc_auc`: ROC AUC
- `macro_F1`: 宏平均F1分数

## 🔄 文件更新机制

- 每个参数值的实验结果会追加到对应的CSV文件中
- 如果文件不存在，会自动创建并写入表头
- 如果文件已存在，会追加新行（不重复表头）

## 📝 分析的参数

脚本会分析以下参数的敏感性：

1. **weight_loss_ms_disc**: 多尺度域对抗损失权重
2. **weight_loss_disc**: 单尺度域对抗损失权重
3. **prototypical_margin**: 原型网络边界
4. **weight_loss_src_sup**: 源域监督对比损失权重
5. **weight_loss_trg_inj**: 目标域注入对比损失权重
6. **scale_weights**: 多尺度层权重

## 🚀 使用方法

运行参数敏感性分析：

```bash
# 运行所有参数的敏感性分析
python experiments/sensitivity_analysis.py --dataset MSL --src F-5 --trg C-1

# 运行特定参数的敏感性分析
python experiments/sensitivity_analysis.py --dataset MSL --src F-5 --trg C-1 --param weight_loss_ms_disc
```

## 📝 注意事项

- 实验结果会自动保存到此文件夹
- 评估脚本会将结果同时保存到 `experiment_results/MSPAD_*.csv` 文件中
- 此文件夹中的CSV文件包含额外的参数信息（`param_name` 和 `param_value`）
- 建议定期备份此文件夹中的结果文件

