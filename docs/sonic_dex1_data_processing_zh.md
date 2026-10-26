# SONIC + Dex1 数据转换与 DiT4DiT 加载说明

本文记录 2026-10-26 对 `block_into_cup_v1_1_dex1` 数据快照的处理。目标是在保留 SONIC v1.1 动作语义的前提下，让 DiT4DiT 学习 **64 维 motion token + 左右各 1 维连续夹爪目标**。本次完成数据转换和原生加载验证，尚未上传 Hugging Face、启动模型训练或验证实机控制。

## 1. 输入、输出与本次结果

原始数据目录：

```text
/home/cqu023/GR00T-WholeBodyControl/outputs/block_into_cup_v1_1_dex1
```

本次输出目录：

```text
/home/cqu023/GR00T-WholeBodyControl/outputs/block_into_cup_v1_1_dex1_dit4dit_20261026_full
```

- 输入共有 41 段，编号 0–40，共 111,768 帧。编号到 40 不代表只有 40 段。
- 输出保留 34 段，共 94,284 帧，50 Hz，约 31.4 分钟，文件合计约 388 MB。
- 输出 episode 连续编号为 0–33，全局帧索引也重新连续编号。
- 原始目录及之前的旧转换目录均未修改或覆盖。新增采集不会自动进入这次输出，需要重新转换到新目录。

## 2. 为什么需要转换

公开的 DiT4DiT 原有机器人配方面向 Decoupled WBC 命令。这里采用独立 SONIC 配方，使 Action DiT 学习 SONIC latent 动作，而不是将 latent 强行转换为腰部姿态、底盘速度等 WBC 命令。

处理主要解决三个问题：使状态与动作字段符合加载器契约；排除无效示教标签；为连续 token 和夹爪值提供正确的归一化与反归一化。模型预测的 token 后续需要接入匹配的 SONIC 解码与实时控制链，格式正确本身不代表机器人已经能稳定完成任务。

## 3. 做了哪些数据处理

### 3.1 先筛选完整、有效的 episode

只处理已登记在 episode 元信息中的完整片段，并检查元信息、Parquet 和视频是否一致。检查包括字段、维度、有限数值、帧数、时间戳、视频尺寸与帧率，以及 Dex1 遥测会话和控制状态。

本次按照既定规则整段排除：

- 原始 episode 0、23、25、28、34：已标记 `discarded`。
- 原始 episode 4：有 1 帧 `dex1.trigger_valid` 无效。
- 原始 episode 7：有 2 帧 `dex1.trigger_valid` 无效。

通用筛选规则还会拒绝包含无效反馈、无效下发目标、缺字段、错误维度、非有限值、遥测会话切换或非 READY 控制状态的片段。trigger 不作为模型动作标签，但本次按约定也将其有效性纳入整段质量筛选。

采集链的 100 ms 阈值用于判断反馈、命令、trigger 各自样本是否过期，不是要求夹爪在 100 ms 内到达目标。转换器读取采集时保存的有效标记，不把无效占位值重新解释成真实标签。

入选的原始片段依次为：

```text
1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
19, 20, 21, 22, 24, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40
```

它们依次对应输出编号 0–33。具体映射及每段帧数写入转换报告，程序不硬编码这些编号。

### 3.2 定义 66 维动作，保留真实动作语义

加载器按以下顺序拼接动作：

```text
[0:64]  action.motion_token[0:64]
[64:65] action.dex1_target_position[0:1]  左夹爪最终下发目标
[65:66] action.dex1_target_position[1:2]  右夹爪最终下发目标
```

- 64 维 token 直接保留源数据数值，不重新编码，也不将某一维解释成单个关节。
- 夹爪目标保留连续弧度值，不做开/闭二值化，不用 trigger 或旧 7+7 手势替代。
- 下发目标是经过控制逻辑处理、DDS 写入接受的最终目标，不等于夹爪已经执行到该位置的确认。
- 夹爪位置反馈作为状态，不能冒充动作标签。
- Parquet 保留原始动作列；通过 `modality.json` 映射及加载器拼接得到 66 维张量，不要求另存一列 66 维数组。

### 3.3 新增 31 维状态，移除旧手部占位对模型的影响

原始 `observation.state` 为 43 维，含身体关节及旧的左右各 7 维手部字段。新增 `observation.sonic_state`：

```text
concat(
    observation.state[0:22],
    observation.state[29:36],
    observation.dex1_position[0:2]
)
```

其顺序为 29 个身体关节，再接左、右 Dex1 实测位置。身体部分依次为左腿 6、右腿 6、腰部 3、左臂 7、右臂 7 个关节，详细关节名保存在转换报告和 `info.json` 中。

原始 43 维状态及遥测列仍保留用于追溯；模型只通过配置读取新的 31 维状态。连续夹爪反馈和目标的区别因此被保留下来。

### 3.4 保留时间线、图像和任务描述

- 每个入选片段完整保留：不删帧、不插值、不压缩时间线。
- 保持原始帧顺序、片段内 `frame_index` 和 `timestamp`，保持 50 Hz。
- `ego_view` 视频直接复制，不重新编码，并核对复制前后的 SHA-256。
- 保留任务文本与任务索引关联；本次任务文件与源数据一致。
- 全身和三点遥操片段均可保留，原始遥操模式和遥测列用于追溯，不自动加入网络输入。
- 三点目标或 SMPL 在某种模式下为零，不影响以有效 `action.motion_token` 为监督目标；它们不是本配方的动作标签。

### 3.5 重建统计并保护输出完整性

按入选数据重新计算状态和动作统计，数据文件内保留原始值，加载时才做 min-max 归一化。这样不会将已排除片段的分布混入训练统计。

转换前后检查源元信息及相关数据、视频文件的指纹；检测到源文件变化则终止，避免采集过程中读到不一致快照。输出先写临时目录，校验完成后发布最终目录。已有目标目录拒绝覆盖，无可用片段时明确失败。

## 4. 输出了哪些文件，各自有什么用

以下路径均相对本次输出目录。

- `data/chunk-000/episode_*.parquet`：保存重新编号后的逐帧数据、新增 31 维状态、原始动作和遥测，用于训练和追溯。
- `videos/chunk-000/observation.images.ego_view/episode_*.mp4`：保存对应片段的原视频内容，供视觉输入使用。
- `meta/info.json`：保存 LeRobot v2.1 版本、字段定义、形状、50 Hz、总帧数、片段数、数据路径、原采集配置和夹爪标定等。
- `meta/modality.json`：定义模型使用哪些字段、各字段切片与拼接顺序，并标明绝对值语义；它决定 31 维状态与 66 维动作如何组成。
- `meta/tasks.jsonl`：保存任务编号与语言描述的关联。
- `meta/episodes.jsonl`：登记每段编号、长度和任务信息，供加载器识别完整片段。
- `meta/episodes_stats.jsonl`：保存逐片段统计；重新计算数值字段统计，源数据若有视频统计则保留。
- `meta/stats_gr00t.json`：保存映射所需状态与动作列的聚合统计，包括 min、max、mean、std、q01、q99，供加载器读取。
- `meta/dex1_quality.jsonl`：按新 episode 编号生成夹爪质量汇总，并保留源编号。入选片段无效计数均为零；这些零是无效帧计数，不是夹爪位置。
- `meta/conversion_report.json`：记录源目录、动作布局、身体关节顺序、标定和模型配置、新旧编号映射、排除原因及源文件快照。
- `meta/source_equality_validation.json`：记录全部 94,284 帧的源数据一致性检查结果。
- `meta/loader_validation.json`：记录原生单数据集、mixture、批处理及归一化往返验证结果。
- `README.md`：供数据集使用者和后续 Hugging Face 页面阅读的字段、筛选和加载说明。

质量日志是额外审计文件，不需要将逐段内容塞入 `info.json`。字段定义及采集配置放在 `info.json`，逐帧有效性在 Parquet，逐段质量汇总在 JSONL，各有用途。训练加载器不会仅因存在质量日志而自动过滤数据；本次过滤由转换程序执行。

## 5. 转换程序与 DiT4DiT 配套代码

### 5.1 GR00T 仓库中的转换入口

- `/home/cqu023/GR00T-WholeBodyControl/gear_sonic/scripts/convert_sonic_to_dit4dit.py`：筛选片段、构造状态、重编号、复制视频、生成元信息和统计，并发布完整输出。
- `/home/cqu023/GR00T-WholeBodyControl/gear_sonic/tests/test_sonic_conversion.py`：覆盖筛选、字段和维度异常、非有限值、源文件变化、已有输出目录等转换行为。

### 5.2 DiT4DiT sonic 分支中的适配

以下代码已在此次文档编写前实现，本次新增的是处理说明文档。

- `DiT4DiT/dataloader/gr00t_lerobot/data_config.py`：注册 `g1_sonic_dex1`，定义状态、动作模态和 min-max 变换。状态不使用默认 sin/cos 展开，保持 31 维。
- `DiT4DiT/dataloader/gr00t_lerobot/mixtures.py`：注册对应 mixture，使训练入口能够选择该数据集。
- `DiT4DiT/dataloader/gr00t_lerobot/embodiment_tags.py`：注册对应 embodiment 映射。
- `DiT4DiT/dataloader/gr00t_lerobot/datasets.py`：保存 SONIC 动作契约、归一化方式和空二值维度列表。
- `DiT4DiT/dataloader/gr00t_lerobot/sonic_normalization.py`：提供 SONIC min-max 反归一化和常量维度恢复。
- `DiT4DiT/model/framework/base_framework.py`：为 SONIC 增加独立反归一化分支，避免原默认逻辑将索引 6，即第 7 维 token 二值化；其他机器人原有行为保留。
- `DiT4DiT/config/real_robot/dit4dit_sonic_dex1.yaml`：独立配置 66 维动作、31 维状态、50 帧动作窗口、图像和数据参数。
- `examples/Real_G1/train_files/run_sonic_dex1.sh`：独立训练入口，通过 `COSMOS_MODEL` 指定已有模型权重，不套用动作维度不匹配的 WBC action checkpoint。
- `examples/Real_G1/validate_sonic_dex1.py`：使用原生加载接口验证形状、图像、文本、尾部填充和反归一化。

归一化类型和空二值维度列表在 DiT4DiT 保存训练统计时加入；不能仅凭数据集 `stats_gr00t.json` 含有 min/max 就假设默认反归一化逻辑正确。

## 6. 加载约定与复现命令

### 6.1 转换

在数据所在主机执行：

```bash
cd /home/cqu023/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate
python gear_sonic/scripts/convert_sonic_to_dit4dit.py \
  --input outputs/block_into_cup_v1_1_dex1 \
  --output outputs/block_into_cup_v1_1_dex1_dit4dit_new \
  --dry-run
```

检查筛选结果后移除 `--dry-run`，执行实际转换。示例使用新目录名；已存在的输出目录即使 dry-run 也会被拒绝，不能直接覆盖本次成果。

### 6.2 数据目录名与配置必须匹配

当前 mixture 和验证脚本默认使用目录名 `block_into_cup_v1_1_dex1_dit4dit`。这次完整输出带 `_20261026_full` 后缀，不能直接沿用旧配置而误读之前的旧转换目录。

本次原生验证使用临时数据根目录，将默认数据集名链接到本次完整输出，并通过临时 YAML 指向该根目录，没有修改仓库原有配置。后续服务器部署时，可将本次完整数据下载到“配置的数据根目录 / `block_into_cup_v1_1_dex1_dit4dit`”；也可以同步修改 mixture 和验证脚本的数据集路径。

当前加载器使用 `lerobot_version="v2.0"` 的 JSONL 兼容分支；磁盘数据的 `info.json` 仍如实标注 v2.1，二者不矛盾，不要为适配加载分支篡改数据版本。

样本约定：

- 状态：`1 × 31`。
- 动作：`50 × 66`，`future_action_window_size=49`；尾部使用末帧绝对动作填充，不跨 episode。
- 视频原始偏移：0–16；采样比例 2，最终每个样本 9 张 `3 × 224 × 224` 图像。
- 状态和动作在加载阶段 min-max 归一化，常量维度反归一化恢复原值。

## 7. 本次验证结果与边界

已完成：

1. 全部 94,284 帧的 64 维 token、左右目标、31 维状态、时间戳、帧索引和任务索引与源数据一致；任务文本一致。
2. 转换器检查视频帧数、尺寸与帧率，并验证复制视频内容哈希一致。
3. 原生单数据集对全部 34 段的首、中、尾位置验证，共 102 个窗口；原生 mixture 和 batch size 2 的批处理读取通过。
4. 样本状态为 `1 × 31`、动作为 `50 × 66`，图像和文本可读取，归一化结果无 NaN/Inf。
5. 动作归一化再反归一化的最大误差为 `4.440892098500626e-16`；验证 token 第 7 维不会被二值化，常量维度可恢复，尾部窗口不跨片段。

验证结论是“数据格式和加载链路通过”。本次没有运行模型训练、模型前向性能验收或 SONIC 实机推理，也没有证明混合遥操数据的最终任务成功率。Hugging Face 上传和训练服务器部署仍是后续步骤。
