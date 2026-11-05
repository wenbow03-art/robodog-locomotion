# 给新 agent 的接手说明

## 当前状态

用户当前要求：整理直行模型和实机接入资料，上传到 Git，使另一个 agent 克隆后能继续。用户说实机地址稍后提供，当前先完成文件准备。

- 选定模型：`lz_q1l_straight_20261105/policy/model700_bands.pt`（相对此文件）。它是完整 TorchScript 组合策略，2.1 MiB 左右，已直接提交 Git。
- SHA-256：`109c87fb85abb565840990497a04aa003632cf164896b332073b9246545a5e30`。
- 此包的推理检查、读取消息工具无需原训练目录、GPU或Isaac Gym。
- 原始服务器验证：Python3.8 / PyTorch1.10.0+cu113，CPU两轮×290步，输出维度、有限性、状态范围及重置后完全一致性通过。新环境验证记录见 [validation](validation)。
- 实机未连接；地址、关节零位/方向、IMU外参、横向定位来源待确认。尚无实机电机控制适配器。
- 历史模型仍在 `models/`，当前接入工作用上述模型。原历史任务记录见 [history](history)，不代表有正在运行的训练任务。

## 克隆后先运行

在仓库根目录：

```bash
python3 deployment/check.py
```

此命令仅用标准库检查部署包中的所有文件散列。模型文件与配置随仓库提供，没有待下载的LFS指针，也不会回原服务器找文件。

只想取实机资料时可做轻量克隆：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/wenbow03-art/robodog-locomotion.git
cd robodog-locomotion
git sparse-checkout set deployment
python3 deployment/check.py
```

## 独立 CPU 推理环境

以下配置用于 Linux x86_64 / Python3.10，检查 TorchScript 接口，不驱动机器人：

```bash
python3.10 -m venv .venv-deploy
.venv-deploy/bin/python -m pip install -r deployment/requirements-cpu.txt
.venv-deploy/bin/python deployment/check.py --model
```

若 Debian/Ubuntu 提示 `ensurepip is not available`，先安装对应的 `python3.10-venv` 系统包再创建环境。环境依赖的实测版本列表保存在 `deployment/validation/cpu-python310-freeze.txt`。CPU软件包从PyTorch官方CPU索引取得，不需要CUDA。

这组依赖不是ARM板端安装方案。板端优先用已有Python/PyTorch运行同一检查，记录架构与加载结果，再决定如何适配；不要把原服务器虚拟环境复制到ARM机器。文件完整性检查不需要PyTorch，ROS2状态采集也不需要PyTorch。

## ROS2 消息环境

采集工具使用ROS2发行版对应的系统Python。已经有SDK的机器人，先source原SDK的 `install/setup.bash` 并确认：

```bash
ros2 interface show robot_common_interface/msg/JointCommand
```

若开发机缺少该消息包，仓库附带其完整原始源码，可以单独编译。以下为ROS2 Humble / Ubuntu22.04的命令；需要已安装ROS2、colcon及ROS消息生成依赖：

```bash
source /opt/ros/humble/setup.bash
cd deployment/ros2_ws
colcon build --packages-select robot_common_interface --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=OFF
source install/setup.bash
cd ../..
/usr/bin/python3 deployment/lz_q1l_straight_20261105/tools/collect_robot_state.py --help
```

随包只带接口定义，没有自动安装/启动驱动节点。消息源码声明 Apache License 2.0，来源和散列见 `deployment/ros2_ws/SOURCE.json`。

## 收到实机地址后的步骤

先读 [部署包README](lz_q1l_straight_20261105/README.md)，按用户实际提供的SSH地址连接。`orangepi@192.168.8.10`仅为说明书默认值；不要把它当作已经确认或已经连接的机器。

1. 上传部署包到新目录，运行完整性检查及板端模型加载检查。
2. 在板端已有ROS2/SDK环境中运行 `tools/inspect_robot.sh`，记录架构、依赖、控制服务和话题发布者。
3. 用 `tools/collect_robot_state.py --duration 10 --output captures/first_state` 采集只读关节、IMU和驱动诊断。
4. 阅读日志，确认关节名称/顺序及单位。完成校准后写入 `calibration.local.json`，未知值保留null。
5. 确定同坐标系、同时间基准下的航向和横向位置来源，再实现观测适配、日志回放验证及后续控制接口。

前述脚本不会停用 `infer_zero.service`、发布 `/joint_actuator_commands` 或使能电机。现场控制测试需要另行确定控制权切换、限位、超时处理和现场配合。不要因为文件或CPU测试通过而把 `hardware_ready` 改成true。

## 关键接口与已知限制

完整定义在 `config/policy_contract.json`：50Hz，最新帧在前的6×45维历史，归一化动作12维，内部已有0.3秒站走融合。输入还包含有符号yaw角速度、航向误差和横向偏移。航向/侧偏基于固定起始直线，IMU单独不能给出横向位置。

策略顺序为RL、RR、FL、FR；SDK为FL、FR、RL、RR，每腿ABAD/HIP/KNEE。按名称重排，SDK内部已应用电机方向/传动系数，勿重复换算。SDK与URDF零位、额外符号和IMU外参尚未标定。

`reference/`中的训练/导出代码与历史manifest用于查来源，可能包含原服务器路径。这些路径不参与部署检查或采集工具运行。URDF只有结构快照，不能据此声称完整仿真可运行。模型的仿真路径偏移和各速度差异保存在 `policy/evidence/`，尚无实机效果证据。

## 可以直接交给新 agent 的提示词

> 先读仓库AGENTS.md和deployment/HANDOFF.md，运行deployment/check.py；按说明建立CPU环境并运行--model。继续准备LZ-Q1L实机接入，使用指定model700_bands.pt，核对来源和接口。机器人地址由我稍后提供；先完成离线检查。收到地址后先做只读检查与状态采集，记录关节零位/方向、IMU外参和横向定位仍缺什么，不把离线检查说成实机验证。

后续结论写入新的现场记录并更新本文件的当前状态。保留原始权重、验证记录和历史结果；凭据、实机原始日志、本地标定及环境目录已列入gitignore。
