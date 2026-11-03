# 延迟扩展交付记录（2026-11-03）

所有开发修改位于229。未修改32，未部署或重启G1服务，未替换正在使用的SONIC二进制。

## 已加入的记录
- 32：WebSocket文本接收完成、JSON解析完成分开记录。单程边界包含协议栈及缓冲，不等于纯网络传播。
- 229 SONIC：token解析、策略Infer调用、命令准备、命令排队、DDS Write调用。
- G1相机候选程序：请求到达、完整MJPEG帧在用户态可用、选帧、sendall调用及返回。
- chrony只读观察器：两端连续有效时钟证据覆盖事件才计算单程耗时，报告携带uncertainty_ms；缺失证据不填零。

## 现有路径与开启条件
原有collect/relay继续使用，32必须先拉取这次代码才能有server_decoded。
请勿用旧install.py的v1快照覆盖已经更新的源文件；该安装器哈希不匹配会拒绝安装。
Python埋点由LATENCY_TRACE_UDP启用。SONIC原生埋点需要用户重新构建并在计划好的停止/启动流程中切换。
原生进程启动前设置：
    export LATENCY_TRACE_HOST=229
    export LATENCY_TRACE_UDP=127.0.0.1:19555
    export LATENCY_TRACE_EVERY=10
默认每10个控制周期采样一次。未设置UDP时关闭。非阻塞UDP可能丢包；启用有额外开销，尚未做实机开销测量。
DDS Write返回不是机器人接收确认，不是实际执行完成。
token请求与SONIC解析可关联；独立decoder周期尚未证明与每个token一一对应。

## 时钟证据
在已经配置好chrony同步的每台参与机器运行（每机单独终端）：
    LATENCY_TRACE_HOST=229 python3 tools/g1_latency_trace/clock_watch.py --output outputs/latency/exp_002/229_clock.jsonl
32将HOST改为32、文件名改为32_clock.jsonl。先启动本机collect；观察器也把证据发送至本机收集器。
只记录，不设置系统时间。229检查时尚未发现chronyc，故目前不能声称跨机单程计时已验证。
G1也必须有时钟证据，才能拆分G1和229之间的单程时间。
结束后对汇总文件执行：
    python3 tools/g1_latency_trace/latency_trace.py report outputs/latency/exp_002/229_combined.jsonl --output outputs/latency/exp_002/report.json
无--sync时自动读取汇总事件内的clock_sync。时钟窗口最长10秒，不外推；失步、过期、重启或时间跳变被拒绝。
误差界限基于chrony提供的UTC参考和健康假设，不能替代硬件时间戳。

## 相机与实际执行边界
G1服务源文件已只读定位到 /home/unitree/pico_usb_stereo/robot_usb_stereo.py，PID2409（检查时）。
其v4l2-ctl直接读取硬件MJPEG，无Python JPEG编码步骤。
准备好的兼容版本在GR00T仓库 tools/g1_camera_latency/，详见该目录README.md。
未测得曝光开始、硬件编码耗时、DDS机器人接收以及电机实际执行；这些不能用本次记录推算为准确值。
实际执行需要控制器确认或带对应序列号的反馈，不能用旁路订阅或Write返回冒充。

## 验证边界
使用假相机帧、localhost UDP、假策略和临时编译产物测试，未启动机器人控制测试。
完整SONIC主源文件编译通过（临时object），没有覆盖现有二进制；尚未实机闭环验证。

最终检查：
- 229 sonic环境RTC测试17项通过（teleop环境缺FastAPI，改用已有sonic环境后通过，没有安装依赖）。
- 计时与时钟测试12项通过。
- 相机兼容及C++遥测测试3项通过。
- 原有joint_target_limiter、joint_target_smoother两个原生测试通过。
- 两仓库git diff --check通过。
- 尚未提交Git，尚未部署到32/G1；没有实机运动或长期推理测试。
