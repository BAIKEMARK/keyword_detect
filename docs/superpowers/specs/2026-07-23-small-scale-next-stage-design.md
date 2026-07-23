# 小规模下一阶段实验设计

## 范围

本轮只实现三项能力：修复 WavLM matcher 在 dev 评估阶段的显存与断点问题，
增加冻结语音 encoder 后的轻量时序 CTC Adapter，以及将 WavLM Large 和
HuBERT Large 下载到服务器持久化目录的脚本。不启动全量训练，也不实现难负
样本和监督融合。

## Matcher 评估与恢复

- 新增 `--eval-bs`，训练 batch 和 dev batch 独立；默认 dev batch 为 32。
- 每个 epoch 完成训练后、开始 dev 前，先将模型头、optimizer、AMP scaler、
  RNG、当前 epoch 和 `evaluation_pending=true` 原子写入 `.last.pt`。
- 如果评估 OOM 或实例中断，从该 `.last.pt` 恢复时先重新评估已训练好的 epoch，
  不重复训练；评估成功后再进入下一 epoch。
- `eval_bs` 记录在 checkpoint 中但允许恢复时修改。

## Temporal CTC Adapter

- 保留现有 `linear` CTC 头作为默认值和旧 checkpoint 兼容路径。
- 新增 `--head temporal`、`--adapter-dim` 和 `--adapter-layers`。
- temporal 头先对冻结 encoder 各层做可学习加权，再投影到较小维度，经过带
  padding mask 的深度可分离时序卷积残差块，最后输出字符或音素 CTC logits。
- encoder 全部冻结，最终 CTC 头由比赛训练数据训练，保持现有赛规边界。
- checkpoint 保存 head 类型和 Adapter 参数；训练、dev 导出和 eval 推理自动
  按 checkpoint 重建对应结构，旧 checkpoint 默认使用 `linear`。

## 模型下载

- 新增脚本调用 ModelScope CLI，将 `microsoft/wavlm-large` 和
  `facebook/hubert-large-ll60k` 直接下载到 `/mnt/workspace/models`。
- 支持单独下载或一次下载全部，已有完整目录直接跳过。
- 下载完成后检查 `config.json` 和 Transformers 权重文件，并打印后续
  `--model-id` 使用路径。

## 验证

- matcher CLI、配置记录、pending-evaluation 恢复路径有单元测试。
- temporal 头验证 padding 不影响有效帧、反向传播有限、encoder 保持冻结。
- 旧 linear checkpoint 的训练和推理构造保持兼容。
- 下载脚本通过 shell 语法检查；完整模型下载只在服务器执行。
