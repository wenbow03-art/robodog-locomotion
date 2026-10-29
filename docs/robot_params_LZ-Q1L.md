# LZ-Q1L 小型四足机器狗参数汇总（赛道二 / 竞速）

整理日期：2026-10-04。下文路径均相对于
`2026年移动-蓝思杯培训资料/2026年移动-蓝思杯培训资料/`，记 `SDK/` = `赛道二开发/02-模型与SDK/解压/`。

可信度标记：
- **[源]** 比赛方文件中直接给出的数值
- **[推]** 由源文件计算或由驱动源码推导，未经实机验证
- **[缺]** 资料中没有，需实测或向组委会索取

---

## 1. 文件清单

| 类别 | 文件 | 内容 | 仿真用途 |
| --- | --- | --- | --- |
| 本体模型 | `SDK/22nm_dog_创客大赛/22nm_dog/urdf/22nm_dog.urdf` | 运动学、质量、惯量、关节限位、网格引用；含 `<mujoco>` 编译标签 | **主模型** |
| 本体模型 | `SDK/22nm_dog_创客大赛/22nm_dog/urdf/22nm_dog.csv` | SolidWorks 导出表 | 仅作参考，与 URDF 有出入（见 §8） |
| 网格 | `SDK/22nm_dog_创客大赛/22nm_dog/meshes/*.stl` | 21 个 STL（视觉 + 碰撞） | 视觉 / 碰撞 |
| 执行器 | `SDK/joint_actuator/actuator_config/quadruped_small17_actuator/actuator_config.json` | 12 个关节名、CAN ID、EtherCAT 端口、电机型号 `lens_kaite` | 关节命名与顺序 |
| 执行器 | `.../quadruped_small17_actuator/actuator_params_config.json` | 各型号电机的位置/速度/力矩/kp/kd 范围 | 力矩与速度上限 |
| 执行器 | `.../quadruped_small17_actuator/joint_config.yaml` | 关节方向/传动系数、软件限位（单位：度） | 方向、减速比、限位 |
| 执行器 | `SDK/joint_actuator/actuator_config/robot_default_config.yaml` | 机型 `small17`、网口 `eth0`、IMU 开关、mujoco/实机开关 | 部署配置 |
| 驱动源码 | `SDK/joint_actuator/src/joint_actuator_node.cpp` | 控制周期、MIT 指令换算、超限保护 | sim2real 对齐依据 |
| IMU | `SDK/imu/robot_sensors/`（YIS320，`/dev/ttyS0`） | `/imu` 话题的内容与单位 | 观测设计 |
| ROS2 接口 | `SDK/robot_common_interface/msg,srv` | `JointCommand`、`SimulationReset` 等 | 仿真/实机接口 |
| 手柄 | `SDK/joystick_drivers/joy/config/joy-params.yaml` | 死区 0.5，自动重发 20 Hz | 遥控 |
| 说明书 | `通用资料/机器狗技术资料/小型四足机器狗产品说明书.pdf`（LZ-Q1L V1.0） | 操作、模式、速度范围、电池 | 速度指令范围 |
| 说明书 | `通用资料/机器狗技术资料/中型四足机器狗产品说明书.pdf` | ROS2 话题完整说明 | **只参考软件接口，硬件参数不适用** |
| 赛制 | `评审标准表.docx`、`【参赛必读】...操作指引.docx` | 评分规则、赛道地形种类、时间节点 | 任务定义 |

**注意**：小狗说明书中“表 1-1 技术参数”和“表 1-2 关节运动范围”在 PDF 里是**空白表**，没有任何数值。

---

## 2. 整机参数

| 参数 | 数值 | 来源 |
| --- | --- | --- |
| 自由度 | 12 个驱动关节（每腿 abad / hip / knee） | [源] URDF、actuator_config |
| URDF 总质量 | 15.266 kg | [推] 各连杆质量求和；**不是实机称重值** |
| 质量分布 | 机身 5.961 kg；头部 1.168 kg（head_shake 0.100 + head_nod 1.069）；每条腿 2.031 kg（roll 0.109 + 大腿 1.711 + 小腿 0.174 + 足 0.036）；两个相机共 0.015 kg | [推] URDF |
| 机身网格包围盒 | x −0.249 ~ 0.268 m，y ±0.102 m，z −0.097 ~ 0.171 m | [推] base_link.stl |
| 机身质心 | (0.0077, 0.0037, 0.0027) m（base_link 坐标系） | [源] URDF |
| 机身惯量 | Ixx 0.01298，Iyy 0.01443，Izz 0.02087 kg·m² | [源] URDF |
| 全零关节角时机身高度 | 约 0.31 m（足端连杆原点在机身下方 0.283–0.288 m，足部网格底面再低约 0.028 m） | [推] URDF 正运动学 |
| 电池 | 42 V 告警，低于 39 V 可能保护断电 | [源] 小狗说明书 |
| 实机质量、质心、电池规格、尺寸 | — | [缺] 说明书表格为空；中型狗的 24 kg / 54.6 V **不能套用** |

---

## 3. 运动学（URDF）

### 3.1 腿部几何

| 项目 | 数值 |
| --- | --- |
| 髋（roll 关节）位置 | x = ±0.1825 m，y = ±0.045 m，z = 0 |
| hip pitch 关节 | 与 roll 关节同原点 |
| 大腿：hip → knee | (−0.1414, ±0.0939, −0.1414) m，xz 平面长度 **0.200 m**，零位时向后下方 45° |
| 小腿：knee → 足端连杆原点 | 前腿 (0.151, ±0.010, −0.142)，后腿 (0.148, ±0.010, −0.145)，长度 **0.207 m**，零位时向前下方 45° |
| 足端横向位置 | y ≈ ±0.149 m |
| 关节轴 | roll：x 轴；hip / knee：y 轴 |

可见全零关节角本身就是一个“膝盖约 90° 弯曲”的站姿。

### 3.2 连杆与关节命名（URDF ↔ SDK）

SDK 名称与 URDF 名称的对应关系是**根据命名推断**的 [推]，需实机逐关节确认。

| SDK 关节（EtherCAT 端口 / CAN ID） | URDF 关节 | URDF 子连杆 |
| --- | --- | --- |
| FL_ABAD_JOINT (1/1) | roll_leg_f_l_joint | roll_leg_f_l_link |
| FL_HIP_JOINT (1/2) | pitch01_leg_f_l_joint | pitch01_leg_f_l_link（大腿） |
| FL_KNEE_JOINT (1/3) | pitch02_leg_f_l_joint | pitch02_leg_f_l_link（小腿） |
| FR_* (1/4–6) | *_leg_f_r_* | 同上 |
| RL_* (2/1–3) | *_leg_b_l_* | 同上 |
| RR_* (2/4–6) | *_leg_b_r_* | 同上 |
| — | pitch03_leg_*_joint（**fixed**） | pitch03_leg_*_link（足端） |

SDK 内部关节顺序（即 `whole_body_joint_names_` 的顺序）：
FL_ABAD, FL_HIP, FL_KNEE, FR_ABAD, FR_HIP, FR_KNEE, RL_ABAD, RL_HIP, RL_KNEE, RR_ABAD, RR_HIP, RR_KNEE。

### 3.3 非腿部连杆（全部为 fixed）

| 连杆 | 相对父连杆的位置 / 姿态 | 备注 |
| --- | --- | --- |
| camera_chest_mounted_link | base 下 (0.241, 0, −0.044)，pitch 0.465 rad | 向下约 26.7° |
| head_shake_link | base 上 (0.2226, 0, 0.1514)，pitch 0.4167 rad | CSV 中是 revolute（±0.698 rad），URDF 改成了 fixed |
| head_nod_link | 与 head_shake 同原点 | CSV 中是 revolute（±0.1745 rad），URDF 改成了 fixed |
| camera_head_mounted_link | 在 base 坐标系中约 (0.346, 0, 0.148) | 合成俯仰角约 0，即水平朝前 [推] |

相机型号、内参和时延均未提供 [缺]。URDF 中**没有 IMU 连杆**，IMU 安装位置和外参未知 [缺]。

---

## 4. 关节限位

| 关节 | URDF 限位 | SDK `joint_config.yaml`（电机侧） | 换算到关节侧 [推] | 仿真建议 |
| --- | --- | --- | --- | --- |
| ABAD (roll) | ±0.733 rad (±42°) | ±60° | ±60° | 用 URDF |
| HIP (pitch01) | ±2.2689 rad (±130°) | ±150° | ±150° | 用 URDF |
| KNEE (pitch02) | −1.0472 ~ +0.8727 rad (−60° ~ +50°) | ±150° | ±100°（÷1.5） | 用 URDF |

- URDF 限位比 SDK 软件限位更严，仿真中以 URDF 为准。
- **SDK 零位与 URDF 零位是否一致没有任何资料说明** [缺]。实机“零位模式”是趴下的姿态，而 URDF 的全零角是站姿，二者很可能有偏置，需要上机标定。
- 实机超限保护：驱动节点每个周期检查**电机侧**位置，只要任一关节超出 `joint_config` 限位，就不再下发 MIT 指令，并按 2000 个周期一次的频率打印错误日志。策略必须保证不越界。

---

## 5. 执行器与控制

### 5.1 电机配置

12 个关节全部是 `type: lens`，`model: lens_kaite` [源]：

| 字段 | 电机侧 | 关节侧：ABAD / HIP [推] | 关节侧：KNEE（×1.5 传动）[推] |
| --- | --- | --- | --- |
| 位置范围 | ±12.5 rad | ±12.5 | ±8.33 |
| 速度上限 | ±30 rad/s | ±30 | ±20 |
| 力矩上限 | ±25（单位应为 N·m，资料未明说） | ±25 N·m | ±37.5 N·m |
| kp 范围 | 0 ~ 500 | 0 ~ 500 | 0 ~ 1125 |
| kd 范围 | 0 ~ 5 | 0 ~ 5 | 0 ~ 11.25 |

以上是**电机配置字段推出的指令上限**，不是持续 / 峰值额定力矩，也没有热降额数据 [缺]。

### 5.2 方向与传动系数（`joint_direction`）[源]

| 关节 | FL | FR | RL | RR |
| --- | --- | --- | --- | --- |
| ABAD | −1 | −1 | +1 | +1 |
| HIP | −1 | +1 | −1 | +1 |
| KNEE | +1.5 | −1.5 | +1.5 | −1.5 |

驱动节点的换算（`joint_actuator_node.cpp`）[源]：
- 下发：`q_motor = q_joint × d`，`kp_motor = kp / d²`，`kd_motor = kd / d²`
- 上报（`/joint_actuator_states`）：`q_joint = q_motor / d`，`dq_joint = dq_motor / d`，`τ_joint = τ_motor × d`

因此 |d| = 1.5 的膝关节相当于 1.5 倍减速，关节侧力矩 ×1.5、速度 ÷1.5。SDK“关节侧”的正方向与 URDF 关节轴方向是否一致**未知** [缺]，需要上机逐关节小幅运动核对。

### 5.3 控制回路（实机）[源]

| 项目 | 值 |
| --- | --- |
| 电机线程周期 | 2 ms（500 Hz），实时线程 |
| 控制模式 | MIT |
| 实际生效的指令 | 只有 `position`、`kp`、`kd`；`velocity` 和 `effort` 的前馈在源码中被注释掉，恒为 0 |
| 等效控制律 | τ = kp·(q_des − q) − kd·q̇（目标速度为 0） |
| 启动默认值 | 节点启动时以当前位置为目标，电机侧 kp = 100、kd = 1 |
| 底层指令话题 | `/joint_actuator_commands`（`JointCommand`），使用前必须停止 `infer_zero.service` |
| 状态话题 | `/joint_actuator_states`（`JointCommand`，kp/kd 字段填 0）；`/joint_actuator_other_data`（温度、在线状态、错误码） |

这与 legged_gym 的 `control_type = 'P'`（无前馈 PD）一致，sim2real 对齐比较直接。

---

## 6. 传感器

| 传感器 | 已知信息 | 未知 |
| --- | --- | --- |
| IMU（YIS320，串口 `/dev/ttyS0`） | 话题 `/imu`（`sensor_msgs/Imu`）：只填四元数（姿态）和角速度（源码从 deg/s 转为 rad/s） | **未填线加速度和 `frame_id`**；安装位置、外参、频率、时延 [缺] |
| 关节编码器 | 位置、速度、反馈力矩（经 §5.2 换算） | 反馈力矩精度 [缺] |
| 足端接触 / 力传感 | 无 | — |
| 相机 | URDF 预留胸部、头部两个安装位 | 型号、内参、时延 [缺] |

对 RL 观测的影响：IMU 可以提供投影重力和机身角速度，但**拿不到机身线速度**，需要像 HIMLoco 那样由估计器推出。

---

## 7. 运动指标与赛道

| 项目 | 内容 | 来源 |
| --- | --- | --- |
| 强化模式速度 | 线速度 −0.6 ~ 1.0 m/s，角速度 ±0.8 rad/s（说明书误写为 m/s），可越障、爬楼梯 | [源] 小狗说明书 |
| 高速模式速度 | 线速度 −1.5 ~ 3.0 m/s，角速度 ±1.0 rad/s，只能在平地使用 | [源] 小狗说明书 |
| 赛道地形 | 直线、弯道、楼梯、斜坡、石子路、仿真草地、障碍区（窄门等） | [源] 操作指引、评审表 |
| 赛道尺寸 | 俯视图标注为“豆包 AI 生成”，没有尺寸；实拍图无标尺 | [缺] |
| 评分 | 完成度 30%（≥3 种特殊地形为优秀）；稳定度 25%（仿真与实机误差 <10% 为优秀）；流畅性 25%；用时 20%（线性计分） | [源] 评审表 |
| 时间节点 | 11 月 16 日提交成果，11 月 27 日决赛实机竞速 | [源] 操作指引 |

---

## 8. 模型文件的已知问题（接入仿真前必须处理）

1. **网格文件名大小写不一致**：URDF 引用的是 `pitch01_leg_b_l_link.stl`，实际文件是 `pitch01_leg_b_l_link.STL`。Linux 区分大小写，加载会失败。
2. **该网格面数过高**：`pitch01_leg_b_l_link.STL` 有约 4.9 万个三角面，其余大腿网格约 3 千个。直接用作碰撞体会明显拖慢仿真，建议改用简化碰撞体或基本几何体。
3. **URDF 中 `effort="0" velocity="0"`**：legged_gym 从 URDF 读取 `torque_limits`（`legged_robot.py` 第 406 行），并在第 504 行按它截断力矩。不修改的话，**所有力矩都会被截断为 0**。需要按 §5.1 填入关节侧上限。
4. **足端是 fixed 关节**：legged_gym 默认 `collapse_fixed_joints = True`，会把 `pitch03_leg_*_link` 合并进小腿，足端刚体随之消失。应在 pitch03 关节上加 `dont_collapse="true"`，并设 `foot_name = "pitch03"`。
5. **含 `world` 连杆和 `floating_base_joint`**：这是为 MuJoCo 写的。Isaac Gym 的根连杆应为 `base_link`，建议另存一份去掉这两项的 URDF。
6. **全部碰撞体使用原始网格**：训练时建议把足端换成球体，大腿、小腿换成胶囊体或长方体。
7. **CSV 与 URDF 不一致**：CSV 中左后大腿的关节原点 y 为 −0.09（URDF 为 0），左后膝的关节原点 y 为 0.1839（URDF 为 0.0939）；头部两个关节在 CSV 中是 revolute，在 URDF 中是 fixed。**以 URDF 为准。**

建议做法：原始文件保持不动，把模型复制一份到仿真工程中再修改，并记录修改内容。

---

## 9. 缺失信息清单（需实测或向组委会索取）

- 实机整机质量、质心（称重）
- 电机持续 / 峰值力矩、热保护阈值、力矩单位确认
- SDK 零位与 URDF 零位的偏置；各关节正方向与 URDF 轴的对应关系
- 实机站立姿态的关节角（“站立模式”下读取 `/joint_actuator_states` 即可得到）
- IMU 安装位置、外参、输出频率、时延
- 赛道各段的尺寸、坡度、台阶高度、地面材质
- 相机型号与标定参数
