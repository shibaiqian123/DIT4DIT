# SONIC v1.1 腕部运动学验证

实现分支：`sonic`。参考点是左右 `wrist_yaw_link` 原点，坐标系为 pelvis 局部系；不是 Dex1 TCP，也不代表实机执行误差。
预测 token 与标签 token 分别经过同一个冻结的 v1.1 `g1_kin` 和纯 PyTorch FK，再比较位置和姿态。
只取每个 token 解码的当前帧（内部索引 0）。SONIC 原始 producer 的 future offsets 从 0 开始。

## 训练接入

`DiT4DiT/config/real_robot/dit4dit_sonic_dex1.yaml` 已启用 `trainer.kinematic_validation`。
沿用原训练启动脚本、`trainer.eval_interval=100` 和日志 backend。评价不会参与反向传播或最佳模型选择。
使用新 run_id 启动训练；不要把过去见过全部 episode 的 checkpoint 当成未见验证集模型。
需要旧流程时显式设 `trainer.kinematic_validation.enabled=false`。

默认 seed=42、按 episode 90%/10% 划分（验证数量向上取整）、每次最多 256 个固定窗口，均匀覆盖 episode。
`max_windows=0` 表示完整验证集。完整评价可能明显延长训练间隔。
验证窗口不从训练 iterator 消耗数据；关闭增强/dropout，逐样本固定随机种子，结束后恢复随机状态和模块 training 标志。
多卡每 rank 执行相同数量的 forward，按 sample_id 去除补齐重复；所有 rank 汇总后计算全局 p95。

每个 run 保存：
- `validation_split.json`：数据目录、episode 长度、训练/验证 ID。
- `kinematic_train_metadata.json`：仅训练 episode 计算的归一化统计，与 split 绑定。
- `dataset_statistics.json`：模型使用的统计；验证使用同一份统计。
- `kinematic_evaluation_metadata.json`：checkpoint/config/骨架/关节顺序指纹和评价口径。

恢复训练需保留原 split 和训练统计清单；不修改源数据的统计文件，不迁移旧 run。
当前只支持单一 SONIC 连续 action64+Dex1(1+1)、state31、50 步窗口；不支持二值夹爪或多 dataset mixture。
旧 run 没有 split 时只能显式走离线诊断，不能补一个 split 后声称泛化验证。

## TensorBoard 标签

- `val/eef_kin_left_mean_cm`、`val/eef_kin_right_mean_cm`
- `val/eef_kin_left_p95_cm`、`val/eef_kin_right_p95_cm`
- `val/eef_kin_left_rot_deg`、`val/eef_kin_right_rot_deg`
- `val/eef_kin_{left|right}_mean_cm_h00_09`、`h10_19`、`h20_29`、`h30_39`、`h40_49`
- `val/token_mse`：前64维归一化 token 的有效元素 MSE，预测值不裁剪，不含 Dex1。
- `val/token_out_of_range_fraction`：归一化预测 token 超过 [-1,1] 的有效元素比例。
- `val/eef_kin_valid_frames`、`val/eval_windows`

运动学解码前按原 SONIC min-max 反归一化行为裁剪 [-1,1]。位置使用欧氏距离×100；姿态使用 SO(3) 测地角（atan2 实现）并转换为度。
末尾重复填充帧不参与统计；一个 token 的64维全部有效时才纳入运动学评价。没有有效帧时明确失败；空 horizon 分组不伪造零值。
均值按有效帧聚合，p95 对全局有效帧计算，不平均各 batch 的 p95。
这些是独立验证指标，不新增含义不明的单一 `val/loss`。

## 独立加载器

```python
from DiT4DiT.evaluation.sonic_kinematics import SonicKinematicDecoder
model = SonicKinematicDecoder(checkpoint_path, sonic_checkout_root, device="cpu")
positions, rotations = model.decode(raw_tokens)  # [B,T,64]
# positions: [B,T,2,3]（米）；rotations: [B,T,2,3,3]，左右顺序
```

仅导入 PyTorch、NumPy、OmegaConf 和标准库。只加载 g1_kin 参数，不加载 encoder/g1_dyn，不初始化 IsaacLab、DDS 或控制器；MuJoCo 只用于验收参考。
checkpoint 必须匹配已验证的官方 v1.1 SHA-256，旁边必须存在匹配 `config.yaml`。
训练配置给出 2×32 token、10 帧、MLP [64,2048,1024,512,512,640]。
先将输出还原为 [10,64]，分离每行58维 command 与6维 heading，再把 command 展平；前290维才是完整10帧关节位置，后290维是速度。
使用官方 IsaacLab body 顺序和匹配 `g1_29dof_rev_1_0.xml`，根平移0、根旋转单位阵。
冻结 FP32 运行，decoder 只初始化一次；指纹不匹配或权重缺失、维度错误直接失败。

## 229 离线评价命令

已创建隔离环境 `.venv_kinematic_eval`，继承现有 dit4dit 推理环境，并补齐数据与 TensorBoard 依赖；没有改动原 conda 环境。
229 的24GB GPU 运行完整模型时需要 `--cpu-offload-text-encoder`。该开关只用于离线 CLI，保持权重和采样算法不变。
Cosmos 的 device 属性已改为以 transformer 执行设备为准，避免文本编码器卸载后误用 meta 设备。

```bash
cd /home/cqu023/DiT4DiT
NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=2 PYTHONPATH=. .venv_kinematic_eval/bin/python \
  -m DiT4DiT.evaluation.evaluate_sonic \
  --run results/sonic_dex1_52eps_joint_bs4_20261027_161659 \
  --checkpoint results/sonic_dex1_52eps_joint_bs4_20261027_161659/checkpoints/steps_19000_pytorch_model.pt \
  --cosmos /home/cqu023/models/Cosmos-Predict2.5-2B \
  --data-root /home/cqu023/GR00T-WholeBodyControl/outputs \
  --output results/sonic_kinematic_validation/real_prediction \
  --max-windows 2 --step 19000 --legacy-diagnostic --cpu-offload-text-encoder
```

这是已经验收通过的2窗口诊断命令。`--max-windows 0` 可完整评价；常规固定评价用256。
新训练有原始 split 的 run 不需要 `--legacy-diagnostic`，CLI 会复用其 split 与统计。
模型采样采用 `predict_action` 内配置的 `num_inference_timesteps`；当前模型为4步，不把被框架忽略的 `num_ddim_steps=20` 宣称为实际采样步数。
旧配置缺少 max_action_dim/max_state_dim 时，从已校验模型 contract 恢复66/31。

成功标记为 `REAL_PREDICTION_AND_TENSORBOARD_OK`，仅在真实预测、指标计算、event 标签和 step 验证完成后写入 `evaluation_result.json`。
失败时先查看 traceback，不原样重复 OOM 命令。

## 本次验收（2026-11-02）

- 指标单元测试5项：已通过，含时间布局、已知位移/角度、p95、分组、空掩码和重复样本。
- 官方 BaseModule 对照：MLP 最大绝对误差0。
- MuJoCo 零位与随机姿态对照：位置最大绝对误差4.76e-8 m，旋转矩阵最大绝对误差2.04e-7。
- 真实数据：当前229的默认 mixture 指向13个episode，11训练/2验证；训练统计直接核对 parquet extrema；尾部窗口有效帧数1；原数据统计未改动。
- 正式训练 DataLoader（含 AccessTrackedConfig）取样：action [50,66]、state [1,31]；通过。
- 两进程 CPU Gloo 与单进程结果一致，包含补齐去重、RNG/模式恢复、MetricLogger 写入及 event 回读。
- 19000-step 真实模型：2窗口、100有效帧，20个 scalar 标签已写入 TensorBoard，step=19000。
- 没有启动长期训练；没有验证完整多GPU DeepSpeed训练周期、完整验证集性能或机器人运行。

验收文件位于 `results/sonic_kinematic_validation/`：`parity.json`、`data_check/verification.json`、`distributed/verification.json`、`real_prediction/evaluation_result.json`。
TensorBoard 目录：`results/sonic_kinematic_validation/real_prediction/tensorboard`。
上述旧模型诊断不代表未见数据泛化成绩，也不代表 TCP 或实机任务成功。
