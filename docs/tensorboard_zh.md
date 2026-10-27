# TensorBoard 训练日志

训练入口支持 `trackers` 选择日志后端。SONIC 通用配置默认启用 `[jsonl, wandb, tensorboard]`；已有服务器 YAML 不会自动更新，需要显式增加 `tensorboard`。

只使用本地日志、不登录 W&B：

```yaml
trackers: [jsonl, tensorboard]
```

同时使用 W&B：

```yaml
trackers: [jsonl, wandb, tensorboard]
```

未配置 trackers 的旧配置保持 W&B 行为；空列表禁用指标后端，但终端输出和 checkpoint 保存不变。仅设置 WANDB_MODE 不会开启 TensorBoard，也不会关闭已选中的 W&B；TensorBoard 不受此环境变量影响。

## 记录内容

仅全局主进程写入，按 `trainer.logging_frequency` 和实际优化步数记录已有标量指标：loss、学习率、epoch、数据耗时、模型耗时及代码产生的评估标量。不会新增评估计算，也不保证多卡指标经过全局平均。

- `results/Checkpoints/<run_id>/tensorboard/`：TensorBoard event 文件。
- `results/Checkpoints/<run_id>/metrics.jsonl`：完整标量日志，每条包含 step 和 metrics。
- 原 `summary.jsonl` 继续用于 checkpoint 步数记录。

每次记录刷新日志；正常结束及训练异常退出时关闭 writer。SIGKILL 等强制终止无法保证清理。每次新实验使用独立 run_id，避免曲线混在一起。已有训练进程不会自动加载这些代码变更。

## 在训练服务器查看

在 dit4dit 环境中启动（安装依赖清单已包含 tensorboard）：

```bash
tensorboard --logdir /share/2025slx/DIT4DIT/results/Checkpoints --host 127.0.0.1 --port 6006
```

在自己的电脑建立 SSH 转发：

```bash
ssh -N -L 6006:127.0.0.1:6006 2025slx@10.246.1.31
```

浏览器打开 http://localhost:6006 。不需要 TensorBoard 账号或外网连接。启动训练后看到 event 文件、步数和曲线，才表示实际训练日志可用。

## 离线验证

```bash
PYTHONPATH=. python -m unittest discover -s tests -p test_metric_logger.py -v
```

测试会写入并读回真实 TensorBoard event 文件，检查步数和数值、JSONL、非主进程不写入，以及 W&B 共存和资源清理。无需启动模型或 GPU 训练。
