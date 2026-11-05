# LZ-Q1L 实机连接准备包

整理日期：2026-11-05。选用用户确认的直行 baseline `model700_bands.pt`。
SHA-256：`109c87fb85abb565840990497a04aa003632cf164896b332073b9246545a5e30`。

本包可用于传输、模型加载检查、读取实机状态。尚未连接或验证实机；工具不下发电机指令，不停止或替换现有控制服务。实机闭环控制适配器仍待关节、IMU、定位接口对齐后实现。包内 `hardware_ready=false` 明确记录这一状态。

## 文件入口

| 路径 | 用途 |
| --- | --- |
| `policy/model700_bands.pt` | 唯一待接入策略，TorchScript，内部已有站立/行走融合 |
| `policy/manifest.json`、`policy/export_manifest.json` | 原始来源与导出记录，保留历史绝对路径 |
| `policy/evidence/` | 既有仿真评测摘要和协议 |
| `config/policy_contract.json` | 输入布局、归一化、动作转换和关节对应关系 |
| `config/calibration.template.json` | 实机关节符号/零位偏置、IMU 外参、定位话题待填项 |
| `config/connection.env.example` | 说明书的默认连接地址；不含密码 |
| `tools/verify_bundle.py` | 完整性检查，可选 CPU 推理检查 |
| `tools/inspect_robot.sh` | 读取架构、Python、PyTorch、服务状态和 ROS2 接口 |
| `tools/collect_robot_state.py` | 订阅关节、IMU、驱动诊断，记录 JSONL 和统计摘要 |
| `reference/` | 训练配置、导出代码、SDK 源文件和原始参数说明快照 |
| `source_inventory.json`、`SHA256SUMS` | 源文件来源、散列和整包完整性清单 |
| `validation/` | 本机验证结果；不代表板端或实机测试 |

`reference/` 用于追溯，不是可直接编译运行的完整 SDK；实机应使用已有 SDK 环境。URDF 快照不带网格，仅用于查关节和训练限位。原训练目录、其他 checkpoint 和录像保持在原位。

## 连接和传输

说明书 §2.1 给出的调试口地址是 `orangepi@192.168.8.10`，电脑位于 `192.168.8.0/24` 同网段，避免占用 `.10`。这是文档默认值，需与当前实机核对。登录凭据在终端交互输入，不放进包内。

从持有部署包的电脑执行，若地址不同替换 SSH 目标：

```bash
ssh orangepi@192.168.8.10
```

在电脑项目根目录，用一个新的目标目录传输准备包：

```bash
ssh orangepi@192.168.8.10 'mkdir -p ~/lz_q1l_deploy && mkdir ~/lz_q1l_deploy/straight_20261105'
scp -r deployment/lz_q1l_straight_20261105/. orangepi@192.168.8.10:~/lz_q1l_deploy/straight_20261105/
```

第一条命令若提示目录已存在，应使用另一个新目录名并同步修改后续路径，保留之前的现场记录。传输后进入机器人上的准备包：

```bash
cd ~/lz_q1l_deploy/straight_20261105
python3 tools/verify_bundle.py
python3 tools/verify_bundle.py --model
```

第一条校验只需要 Python 标准库；第二条需要板端已有 PyTorch。已验证的本机环境为 Python 3.8 / PyTorch 1.10.0+cu113；这不规定 ARM 板端的 CUDA 或安装版本，板端兼容性以实际加载结果为准。

## 读取实机状态

在机器人终端先加载它已有的 ROS2 和 SDK 环境。按实际路径执行 `source /opt/ros/<发行版>/setup.bash` 和 `source <SDK工作空间>/install/setup.bash`；通过 `ros2 interface show robot_common_interface/msg/JointCommand` 确认消息已安装。保持原来的 `ROS_DOMAIN_ID`。

```bash
mkdir -p captures
bash tools/inspect_robot.sh > captures/robot_inventory.txt 2>&1
python3 tools/collect_robot_state.py --duration 10 --output captures/first_state
```

重复采集时换一个输出目录名。采集器订阅 `/joint_actuator_states`、`/imu`、`/joint_actuator_other_data`，使用传感器 QoS，记录原始消息、接收时间、消息数、接收频率、最大间隔和最后消息年龄。它检查 12 个关节名称/数组和 IMU 四元数的基本有效性；缺少关节或 IMU 消息、非法样本、手动中断会返回非零退出码。驱动诊断缺失通过消息数 0 标明。频率统计包含发现时间，且是接收端统计，不能代替传感器时间同步验证。成功退出只说明收到结构有效的数据，不能证明关节坐标或姿态外参正确。

现场日志应另行保存/回传 `captures/`。本包没有执行以上 SSH、传输或采集操作。

## 接口约定

推理频率固定 50 Hz，不能直接按驱动 500 Hz 调用，否则内部 15 步融合的时间尺度会改变。

```python
policy = torch.jit.load("policy/model700_bands.pt", map_location="cpu").eval()
# history: float32 [1,270]，6 帧，每帧45维，最新帧在最前。
# errors: float32 [1,3]，机身yaw角速度、相对起始航向角、相对起始直线的横向偏移。
# ids: int64 [1,14]，初始全1；alphas: float32 [1,2]，初始全0。
with torch.no_grad():
    action, ids, alphas = policy(history, errors, ids, alphas)
```

每帧依次为：速度命令 `[vx,vy,wz]×[2,2,0.25]`、机身角速度 `×0.25`、投影单位重力、关节角相对默认姿态 `×1`、关节速度 `×0.05`、上次实际执行的归一化动作。旧历史保持真实，切换速度段时不要重建历史。控制器重新初始化时重置全部状态。零速时 `errors=0`；运动时三个量使用有符号的 rad/s、rad、m，不做观测角速度的 0.25 缩放。

每条腿内部为 ABAD、HIP、KNEE；策略顺序为 **RL、RR、FL、FR**，SDK 原始顺序为 **FL、FR、RL、RR**。必须按名称对应：SDK→策略索引为 `[6,7,8,9,10,11,0,1,2,3,4,5]`。名称映射依据源码推断，机械方向与零位仍需现场校准。

模型输出是归一化动作。训练转换为 `q_des_urdf = q_default + 0.25 × action`，训练默认角度全部为 0，训练动作裁剪 ±100；这个宽裁剪范围不是实机安全限位。训练 PD 为 kp=80、kd=2，仿真物理步长 1ms、每20步推理一次。它们是训练设置，尚未成为已验证的实机参数。

SDK 已在驱动内部完成 `joint_direction` 的电机方向和传动换算；订阅状态是 SDK 关节侧角度，不能再乘除一次该系数。SDK 关节侧→URDF 的额外符号/偏置另填在 calibration 文件中，未知项保留 null。实际下发前还需实现并验证关节限位、命令时效和控制权切换。

航向和侧偏必须对应同一条固定参考线：
`heading=wrap(yaw-yaw0)`，`lateral=-sin(yaw0)*(x-x0)+cos(yaw0)*(y-y0)`。
IMU 只提供姿态与角速度，不能直接提供横向位置。本策略的中速修正使用侧偏输入，需要确认现场的定位/里程计来源；持续把侧偏填零会改变控制器输入含义，不是等价部署。

## 连接后还需落实

1. 核实实机地址、架构、PyTorch 可加载性、SDK 环境和话题发布者。
2. 从只读记录核对 12 关节名称、实际站姿角度、IMU 姿态与轴向；补全关节符号/零位、IMU 外参。仅凭一次静态角度不足以完成全部标定。
3. 确定航向/横向位置来源及其时间戳、坐标系；再完成观测组装和离线日志回放验证。
4. 对齐现有控制服务、限位和超时处理，再安排有现场配合的控制测试。此包只查询 `infer_zero.service` 状态，未包含自动停服或电机启动脚本。

当前模型用于直行，命令 vy、wz 应为0。既有仿真存在速度相关的路径偏移；评测摘要留在 `policy/evidence/`，不能把“未跌倒”当作实机路径跟踪通过。
