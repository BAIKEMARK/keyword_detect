# 全量 WavLM 音频匹配设计

## 目标

让现有冻结 WavLM 帧级匹配模型使用官方 50 万 pair 训练，并具备 epoch 级断点
续训能力。该分支使用 enroll/query 音频和训练标签，作为字符/音素 CTC 的互补
声学证据，不替换当前 CTC 主模型。

## 接口

- `--train-csv` 和 `--train-zip` 选择训练数据，默认值保持原 5 万子集兼容。
- `--subset 500000` 显式选择全量 50 万 pair。
- `--resume` 从已完成 epoch 的下一轮继续；`--epochs` 表示目标总轮数。
- `--out` 保存最佳模型，默认 `<out stem>.last.pt` 保存最近完整 epoch。

## Checkpoint

新 checkpoint 保存匹配头、optimizer、AMP scaler、随机数状态、当前/最佳 epoch
和完整训练配置。恢复时允许修改总轮数、worker、设备和日志频率，但拒绝模型、
数据、batch size、学习率、损失权重或增强配置不一致的续训。

旧 checkpoint 缺少 optimizer 等字段时恢复模型头、epoch 和最佳 AUC，并明确
提示重建训练状态。

## 验证

- CLI 支持全量路径、pair 数和恢复参数。
- 旧 checkpoint 可兼容，新 checkpoint 配置冲突会报错。
- 现有帧级匹配、数据增强、冻结 encoder 和 CTC 测试继续通过。
