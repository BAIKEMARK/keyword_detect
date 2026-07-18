# WavLM CTC 断点续训设计

## 目标

为 `baseline/train_wavlm_ctc.py` 增加 epoch 级断点续训，并在 checkpoint
中保存足够的训练配置，避免再次依靠混合日志反推 batch size。现有推理脚本和
`--out` 最佳模型语义保持不变。

## 命令与文件

- 新增 `--resume CHECKPOINT`，从 checkpoint 中已完成 epoch 的下一轮继续。
- `--epochs` 表示目标总轮数。例如 checkpoint 已完成 epoch 2，传入
  `--epochs 5` 时训练 epoch 3、4、5。
- `--out` 继续保存 Dev Mean AUC 最佳的 checkpoint。
- 新增 `--last-out`。未指定时由 `--out` 派生为 `<stem>.last.pt`，每个完整
  epoch 后都保存，用于可靠续训。

## Checkpoint 内容

新 checkpoint 保存模型头、optimizer、AMP scaler、当前 epoch、当前和历史最佳
dev 指标、随机数状态，以及模型、词表、训练数据、utterance 数量、batch size、
学习率、增强和音频截断等实际配置。写入采用临时文件加原子替换，避免中断留下
半个 checkpoint。

恢复时校验会影响模型或训练语义的关键配置；允许修改目标总轮数、日志频率、
worker 数和运行设备。旧 checkpoint 缺少 optimizer 和 scaler 时，加载模型头、
历史最佳 AUC 和 epoch，以新 optimizer 从下一轮继续，并明确打印兼容模式提示。
这能利用现有全量 epoch 2 模型，但不声称恢复 epoch 3 中途的优化器状态。

## 验证

- CLI 能解析 `--resume` 和 `--last-out`。
- `.last.pt` 文件名派生正确。
- 新 checkpoint 的 batch size 或数据规模不匹配时拒绝恢复。
- 旧 checkpoint 缺少新字段时仍可兼容加载。
- 原有 WavLM CTC 单元测试继续通过。
