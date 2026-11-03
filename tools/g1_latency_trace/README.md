# G1 / 229 / 32 分段延迟诊断

版本管理位置：DiT4DiT 仓库 `tools/g1_latency_trace`；下文命令使用已分发到两台服务器的 `outputs/g1_latency_trace` 副本。新 checkout 可将整个工具目录复制到该位置，或将命令中的 `outputs` 替换为 `tools`。

此工具只记录诊断信息，不发送运动指令，不调整 RTC horizon、过期判断或看门狗。

## 已核实的真实链路

G1 `192.168.123.164:15556` 提供按请求返回的 JPEG；229 上 `USBSource` 发 `N` 请求并解码。
229 的 `RecordingPublisher` 裁右眼、缩放、编码成 JPEG，发布本机5555。
RTC 客户端取图，转 PNG/base64，再经本机8014的 SSH 隧道送32。
32 做模型推理，经隧道回229；RTC客户端向本机5556发ZMQ token。
SONIC C++也运行在229，解码后通过DDS向G1发送关节控制。

现有相机 `timestamps.ego_view` 是229接收/解码时间，不是G1曝光时间。
不能把原日志 Image latency 当作G1到229网络耗时。

## 已接入与测量边界

- `camera_request_roundtrip_229`：发出N请求到收到完整JPEG。包括网络往返、G1等待/取帧/编码，不能称为纯网络单程。
- `camera_jpeg_decode_229`：229端JPEG解码与尺寸检查。
- `observation_build_229`：RTC取缓存图、状态处理、缩放、PNG/base64构造。未单独拆出RecordingPublisher的裁剪/二次JPEG编码。
- `observation_wait_229`：构造好的观测等待被选作推理输入。
- `request_queue_229`、`request_json_229`：发送队列和JSON序列化。
- `request_send_api_229`：WebSocket发送API耗时；返回不等于远端已收到。
- `request_roundtrip_229`：请求发送到完整响应抵达，包含SSH隧道及远端服务。
- `server_wait_32`：32端接收/JSON解析完成到预测线程开始，含排队与前缀准备。
- `preprocess_32`：输入处理及原有的CUDA同步等待。
- `model_and_postprocess_32`：模型、RTC采样、CPU拷贝、反归一化及原有CUDA同步。不是纯GPU kernel时间。
- `predict_total_32`：上述预测线程段总耗时。
- `server_reply_prepare_32`：输出转换、回复准备到调用发送前。
- `reply_json_229`、`reply_accept_229`：响应解析与校验接收。
- `action_queue_229`：接受动作块到某一帧开始发送；后续帧自然会排队，不能全部理解为异常延迟。
- `transport_and_serialization_roundtrip`：229总往返减32服务区间。无需跨机对时，仍包含JSON、隧道、传输及发送调度，**不是纯网络时延**。

脚本预留了 `sonic_received`、`dds_write`、`g1_action_received`、`g1_action_used` 事件，
但此次未修改/编译SONIC C++或G1固件，相关末段显示missing_event，不能宣称测到电机执行。
G1的SSH目前关闭连接，曝光时间、G1编码时间和DDS到达/使用时间尚未接入。
相机RPC使用独立trace ID，其分布可测；尚未实现曝光帧贯穿到动作的全链路ID。

## 安装：不启动任何推理或机器人进程

以下假定工具分别放在两台机器的 `outputs/g1_latency_trace`。
源代码已按2026-11-03读取的版本生成。安装器先核对所有目标文件SHA256；
版本不同直接退出，不会猜测套用。32使用其自己的适配器版本。
在受影响程序停止后安装。执行前可去掉`--apply`只查看。

229：
```bash
cd /home/cqu023/GR00T-WholeBodyControl
python3 outputs/g1_latency_trace/install.py --role 229 --root "$PWD" --apply
```

32：
```bash
cd /share/2025slx/DIT4DIT
python3 outputs/g1_latency_trace/install.py --role 32 --root "$PWD" --apply
```

恢复：同一条命令增加`--restore`。首次安装的原文件另存为`.before_latency`。
SDK被复制到仓库根和部署目录；不依赖ROS，不要求修改现有虚拟环境。

## 使用：建议先不用ROS，直接确认分段数值

32另开采集终端：
```bash
cd /share/2025slx/DIT4DIT
LATENCY_TRACE_HOST=32 python3 outputs/g1_latency_trace/latency_trace.py collect \
  --output /tmp/latency_32.jsonl
```

229另开采集终端：
```bash
cd /home/cqu023/GR00T-WholeBodyControl
LATENCY_TRACE_HOST=229 python3 outputs/g1_latency_trace/latency_trace.py collect \
  --output /tmp/latency_229_combined.jsonl
```

229再开一个只转发诊断日志的终端（不会读取/改动机器人指令）：
```bash
cd /home/cqu023/GR00T-WholeBodyControl
ssh -T 2025slx@10.246.1.32 'tail -n 0 -F /tmp/latency_32.jsonl' | \
  python3 outputs/g1_latency_trace/latency_trace.py relay
```

在229启动相机observer和RTC客户端的**各自终端**加入：
```bash
export LATENCY_TRACE_HOST=229
export LATENCY_TRACE_UDP=127.0.0.1:19555
```

在32启动推理服务器的终端加入：
```bash
export LATENCY_TRACE_HOST=32
export LATENCY_TRACE_UDP=127.0.0.1:19555
```

然后使用原有启动命令；仍由用户控制硬件流程。不必修改horizon。
已运行的Python进程不会因为改文件/另开终端export自动启用埋点。
若observer复用已经存在的5555相机，原相机进程未启用埋点时不会有camera RPC事件。
不要为了诊断直接再启动第二个RTC客户端或占用相同端口的采集器。

采集终端每5秒显示最近最多10000事件的mean/P95/max。
原始JSONL完整保存；UDP诊断是尽力而为，不能作为无丢包的实时控制通道。

离线完整报告（没有转发也可先把32的JSONL复制到229再一并输入）：
```bash
python3 outputs/g1_latency_trace/latency_trace.py report \
  /tmp/latency_229_combined.jsonl --output /tmp/latency_report.json
```
所有时间单位ms。重复event_id会去重；同trace同stage发生歧义则不计算。
缺失、不合法、未对时都不会伪装为0ms。

## ROS 2 接口

32不需要ROS，Noetic不是ROS 2；不混用Noetic和Foxy/Humble环境。
在具有Humble的诊断主机上使用系统Python（不要使用推理venv）：
```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=223
export ROS_LOCALHOST_ONLY=1
LATENCY_TRACE_HOST=229 /usr/bin/python3 outputs/g1_latency_trace/latency_trace.py collect \
  --ros --output /tmp/latency_229_combined.jsonl
```
这条替代前面的229采集命令，不能二者同时占用19555。
32事件仍用上述SSH日志转发进入本地采集器，不要求32有DDS或ROS。
另开终端、同样source和域设置：
```bash
/usr/bin/python3 outputs/g1_latency_trace/latency_ros2_monitor.py
```
诊断话题 `/g1_latency/events`，类型 `std_msgs/msg/String`，best-effort。
话题发布携带原始事件时间，而不是用诊断消息抵达时刻代替真实收发时刻。
本次已验证本机Humble UDP→JSONL→ROS2发布/订阅；Foxy和远程实机尚未验证。

## 跨机器单程时间

仅同一host、同一boot的monotonic时间可直接相减。跨机器不可相减monotonic。
未对时默认显示unsynchronized，不假设网络对称，也不拿RTT/2冒充单程。
若需要分开229→32与32→229，需要记录两端相对共同参考的时钟偏差及误差界。
可通过chrony/PTP管理，但工具不自动修改系统时间，也不安装时间服务。
`report --sync clocks.json`支持：
```json
{
  "229": {"offset_ms": 0.0, "uncertainty_ms": 0.5,
          "valid_from_ns": 0, "valid_until_ns": 1},
  "32": {"offset_ms": 0.0, "uncertainty_ms": 0.5,
         "valid_from_ns": 0, "valid_until_ns": 1}
}
```
这里的值只是格式示例，必须替换成实际校准；有效期也必须覆盖采集事件。
offset定义为本机实时时钟减共同参考时钟；每条结果附两端误差界之和。
SSH/WS发送与接收边界还包含序列化、调度，校准后仍不叫纯链路传播时间。

## 参考项目与验证

本实现为按当前混合传输链路编写的独立实现，借鉴事件关联与源时间戳原则，
没有把ros2_tracing宣称为可以自动解析WebSocket/ZMQ的工具。

- ROS 2官方追踪：https://github.com/ros2/ros2_tracing
- 分布式消息因果链实验：https://github.com/christophebedard/ros2-message-flow-analysis
- ROS 2 Topic Statistics说明：https://docs.ros.org/en/humble/Concepts/Intermediate/About-Topic-Statistics.html

本地测试：
```bash
python3 -m unittest discover -s outputs/g1_latency_trace -p 'test_*.py' -v
/usr/bin/python3 outputs/g1_latency_trace/smoke_ros2.py
```
不在机器人上运行smoke脚本来证明硬件性能；它仅验证本机诊断通路。
