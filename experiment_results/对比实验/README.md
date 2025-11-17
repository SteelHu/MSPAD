# 实验结果汇总文件夹

## 📋 说明

此文件夹用于统一保存所有实验的汇总结果CSV文件。

## 📁 文件命名规则

### MSPAD方法
- `MSPAD_MSL_{src_id}.csv` - MSL数据集结果
- `MSPAD_SMD_{src_id}.csv` - SMD数据集结果
- `MSPAD_Boiler_{src_id}.csv` - Boiler数据集结果

### DACAD方法
- `DACAD_MSL_{src_id}.csv` - MSL数据集结果
- `DACAD_SMD_{src_id}.csv` - SMD数据集结果
- `DACAD_Boiler_{src_id}.csv` - Boiler数据集结果

### CLUDA方法
- `CLUDA_MSL_{src_id}.csv` - MSL数据集结果
- `CLUDA_SMD_{src_id}.csv` - SMD数据集结果
- `CLUDA_Boiler_{src_id}.csv` - Boiler数据集结果

## 📊 CSV文件格式

每个CSV文件包含以下列：
- `src_id`: 源域ID
- `trg_id`: 目标域ID
- `best_f1`: 最佳F1分数
- `best_prec`: 最佳精确率
- `best_rec`: 最佳召回率
- `best_thr`: 最佳阈值
- `avg_prc`: 平均精度（AUPRC）
- `roc_auc`: ROC AUC
- `macro_F1`: 宏平均F1分数

## 🔄 文件更新机制

- 每个源域的所有目标域结果会追加到同一个CSV文件中
- 如果文件不存在，会自动创建并写入表头
- 如果文件已存在，会追加新行（不重复表头）

## 📝 注意事项

- 此文件夹中的文件会在运行评估脚本时自动生成
- 建议定期备份此文件夹中的结果文件
- 如果需要清理旧结果，可以直接删除对应的CSV文件

