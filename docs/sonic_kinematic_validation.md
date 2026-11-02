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

## 显式 token 适配与命名解码输出（第2步）

`SonicKinematicDecoder.to_internal_tokens(raw_tokens)` 将 `[B,H,64]` 还原为 `[B,H,2,32]`。
参数来自匹配配置：`token_dim=num_fsq_levels=32`、`max_num_tokens=2`，并验证 `token_total_dim=64` 和 BaseModule 的输入/输出时间维。
轴对应关系为 `internal[b,h,n,d] = raw_tokens[b,h,n*32+d]`；不交换 B/H，不转成 `[B,H,32,2]`，支持非连续张量。
输入为反归一化后的 SONIC token，不再次编码或 FSQ 量化；现有评价器先反归一化再进入该接口。

```python
internal = model.to_internal_tokens(raw_tokens)  # [B,H,2,32]
fields = model.decode_fields(raw_tokens)
command = fields['command_multi_future_nonflat']        # [B,H,10,58]
heading = fields['motion_anchor_ori_heading_mf_nonflat'] # [B,H,10,6]
positions, rotations = model.decode(raw_tokens)          # 原腕部评价接口不变
```

`sonic_token_adapter.py` 移植了官方 `UniversalTokenModule.decode()` 的按配置拼接输入、按有序字段维度拆分输出逻辑。
运行时不导入完整 SONIC 训练模块。输出字段顺序来自 `decoders.g1_kin.outputs`；宽度来自已核实的 observation producer 契约：command=2×29 DOF，heading=6D旋转。
MLP 只展平最后两个 token 轴；输出恢复时间维后才按字段拆分。关节角从命名 command 字段的 qpos/qvel 布局恢复，不再切片原始 MLP 输出的固定前缀。

顺序证据：
- `gear_sonic/utils/inference_helpers.py` 的 encoder ONNX 导出使用 `selected_tokens.flatten(start_dim=-2)`。
- `UniversalTokenModule.forward()` 对内部 token 使用同一最后两轴展平规则；`assemble_all_tokens()` 组织为 `[B,S,num_tokens,token_dim]`。
- 部署 C++ 的 `GatherTokenState()` 按相同索引复制 encoder buffer。
- `run_data_exporter.py` 将 `proprio['token_state']` 原序写入 `action.motion_token`。
- `convert_sonic_to_dit4dit.py` 保留该列，且本次逐行对比源/目标 parquet 全部38,333行一致。

新增回归覆盖非连续输入、B/H/token/feature索引、错误宽度、非有限值、配置字段顺序及输入字段顺序。
验收脚本直接执行官方源文件的 `assemble_all_tokens` 和 `decode` 方法，并调用官方 `BaseModule`：随机token及真实记录token逐字段一致。
本次14项测试通过，官方命名字段最大差0，FK对照仍通过。结果：
`results/sonic_kinematic_validation/token_adapter_parity.json`。
本次第2步不重复执行完整 GPU 策略预测；验证的是实际 checkpoint 的解码与数据顺序，前一节的真实策略/TensorBoard验收属于第1步。

### Reconstruction fields, selected links and optional TCP (step 3)

`decode_reconstruction(tokens)` returns `joint_positions` and `joint_velocities`
`[B,H,10,29]` in IsaacLab order, plus the untouched heading first-two-column
representation `[B,H,10,6]`. The named command field must be flattened and split
into ALL positions followed by ALL velocities, not split each 58-wide row.
This matches the temporal-bug workaround in official
`decoder_output_to_egocentric_transforms`. The official `G1JointPositionLoss`
and `G1JointRotationLoss` exist and call that helper; its generic old `ori_b`
field gate cannot directly parse the matched v1.1 `ori_heading` output.

`decode(tokens)` still selects frame zero and returns `[B,H,2,3]` positions
and `[B,H,2,3,3]` matrices. FK uses body names to map IsaacLab joint slots to
MJCF tree joints, equivalent to the conversion script's `dof_mj = q[:,MJ_TO_IL]`.
Root translation is zero and quaternion is identity **wxyz**. Heading is not
used to reconstruct world trajectories. Default link origins are left/right
`wrist_yaw_link`, not Dex1 TCP.

`trainer.kinematic_validation.eef_links` selects named left/right MJCF bodies.
Optional `link_to_tcp` must contain both `left` and `right`, each with
`translation_m: [x,y,z]` and `quaternion_wxyz: [w,x,y,z]`. The quaternion must
be unit length. Translation is TCP origin in the selected link coordinates;
rotation maps TCP axes into link axes. Composition is
`p_tcp = p_link + R_link @ translation_m`, `R_tcp = R_link @ R_link_tcp`.
No Dex1 calibration is supplied or assumed. Set `link_to_tcp: null` for link
origins. Use a separate output/log directory when changing link/TCP definitions.
Metadata records selected links, explicit transforms, root convention and
`eef_reference`; the same configuration is used for predicted and label tokens.
Offline evaluation accepts this through its existing `--evaluation-config`.

Step 3 validation: all 17 unit tests passed. Real v1.1 output field parity
remains exact; 38,333 source/export token rows agree. MuJoCo wrist position
max difference: 4.7552e-8 m; rotation matrix: 2.0403e-7. Selected wrist-roll
links and synthetic TCP offsets: max component difference 2.8516e-7.
Report: `results/sonic_kinematic_validation/reconstruction_fk_parity.json`.
Default decode output is exactly unchanged. Full DiT prediction was not
rerun for this step. Synthetic offsets are not hardware TCP calibration.

## 步骤 4–7：完整采样、轨迹日志与可微训练损失

### 采样与统计口径

实际 SONIC 路径为 `rtc_deploy/model_adapter.py:Policy.predict(previous=None)` →
`DiT4DiT.predict_action` → `FlowmatchingActionHead.predict_action`，完成配置规定的
所有 Euler 步后返回归一化 `[B,50,66]`。评价器调用相同接口，移除了不起作用的
DDIM 参数。反归一化复用部署的 `unnormalize_sonic_actions`，每个预测/标签仅一次，
然后截取前 64 维。ZMQ 示例会截取 23 维并对 state 做 sin/cos，属于另一机器人
接口，不能用于 SONIC。本评价不带 RTC 历史动作引导，不宣称验证连续 RTC 闭环。

新训练仍按 episode 保存/复用 90/10 划分，只用训练 episode 的统计。
训练和验证样本现在都携带 `valid_time_mask[H]`、`decoder_reference_mask[H,1]`、
episode 和起始帧。前者排除动作 chunk 尾部重复，后者显式限定当前 SONIC 参考帧。
末端要求全部 64 个分量有效，并取两个时间 mask 的交集；token MSE 使用各分量
有效 mask。decoder 内部未来 1–9 帧不参与本版指标/损失，也不根据 action chunk
步长猜测其有效性。

新增 `val/eef_kin_{left|right}_mean_cm_ref00` 标量，明确参考帧 0；原有五组
horizon、mean、P95、角度、token MSE 等保留。当前共有 22 个标量（全 horizon
组均有效时）。`val/eef_kin_trajectory` 为固定有效样本的左右 xyz 预测/标签曲线，
单位厘米、pelvis 局部系，padding 以断线处理。主进程同时保存
`eef_trajectory_step_<step>.png`，离线入口会读回 TensorBoard 的图像标签和步数。
绘图依赖安装命令：`.venv_kinematic_eval/bin/pip install -r tests/requirements-sonic-eval.txt`。

### 训练损失（默认关闭，显式启用）

在现有训练启动命令追加：

```text
trainer.eef_training_loss.enabled=true
trainer.eef_training_loss.position_weight=1.0
trainer.eef_training_loss.rotation_weight=0.1
```

同时保持 `trainer.kinematic_validation.enabled=true`；仅支持含 action 的训练模式。
权重为初始配置值，未证明能提升任务成功率。关闭时仍返回原 flow loss，保持旧
checkpoint 的参数结构；冻结 decoder 不加入模型 checkpoint 或优化器。

当前 flow 参数化为 `x_t=(1-t)*action+t*noise`、`v=noise-action`，因此辅助损失
使用 `action_clean_hat=x_t-t*v_pred`，不是直接将 velocity 当动作，也不是把完整
验证采样循环放进训练。这个单步干净估计与完整采样结果有不同用途。

`SonicKinematicDecoder.decode_differentiable()` 保留输入梯度；预测分支没有
`no_grad`/`inference_mode`，decoder 参数仍冻结、eval、FP32。目标分支关闭梯度。
先筛选有效帧，再将预测/标签各反归一化一次并经过同一个 decoder 与可微 FK。
损失为左右末端欧氏距离平方的均值（m²）和 SO(3) 测地角平方的均值（rad²），
按配置权重加入原总 loss。日志为 `train/eef_position_m2`、
`train/eef_rotation_rad2`、`train/eef_weighted`。验证指标的厘米/度单位不变。
裁剪沿用部署的 [-1,1]，饱和分量的梯度为零，没有使用 straight-through 近似。
Dex1 后两维不进入该辅助损失。最佳 checkpoint 选择规则未改。

### 本次验收

- 23 项单元测试通过，包括实际 framework/trainer 方法的启用/关闭接线、重复
  diffusion batch 的时间 mask、空 mask、恒等旋转梯度、冻结参数和 Dex1 排除。
- 真实数据划分/训练统计/尾部 padding 回归通过，源统计文件未修改。
- 两进程 CPU Gloo 汇总与串行一致，去重、RNG/模式恢复及 TensorBoard 验证通过。
- 19000-step 模型真实完整采样：2 窗口、100 有效帧，22 个 scalar 和轨迹 image
  的 event 标签/step=19000 检查通过。左右位置均值约 2.7200/2.6766 cm，
  P95 约 4.8382/5.2246 cm，姿态均值约 13.0425/10.8222 度。
- 捕获真实图像产生的 backbone 特征，用真实动作头执行一次辅助损失 backward：
  EEF 对 action decoder 权重的梯度范数 0.15793，冻结 decoder 参数无梯度，
  Dex1 的 EEF 梯度为零；总梯度有限。**优化器更新次数为零**。

结果目录 `results/sonic_kinematic_validation/step47_real_prediction/`：
`evaluation_result.json`、`training_gradient_verification.json`、TensorBoard events
和轨迹 PNG。此旧模型未按本次 holdout 训练，结果仍是诊断指标，不是未见 episode
泛化性能。没有启动长期训练，没有硬件/任务成功率验证。

复现完整采样和一次梯度验证：使用前文离线命令的相同参数，将
`-m DiT4DiT.evaluation.evaluate_sonic` 替换为
`tests/verify_sonic_real_prediction.py`，输出目录改为
`results/sonic_kinematic_validation/step47_real_prediction`。
该测试捕获真实采样的条件特征，再单独验证动作头反向，不启动优化器。
