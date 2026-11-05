# LZ-Q1L 机器狗：实机接入与训练归档

当前入口是 [新 agent 接手说明](deployment/HANDOFF.md)。2026-11-05 选定的直行 baseline 已随仓库保存，克隆后可以直接校验、离线推理，并准备只读 ROS2 状态采集。

```bash
git clone https://github.com/wenbow03-art/robodog-locomotion.git
cd robodog-locomotion
python3 deployment/check.py
```

完整性检查仅需 Python 标准库。CPU 模型检查、环境安装和只读采集步骤见 [HANDOFF](deployment/HANDOFF.md)。部署路径无需 GPU、Isaac Gym 或 Git LFS；实机尚未接入，尚未完成标定和电机控制适配。

| 入口 | 内容 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 新 agent 首读规则、当前模型、下一步 |
| [deployment/HANDOFF.md](deployment/HANDOFF.md) | 当前状态、可执行命令、缺失信息、交接提示词 |
| [部署包](deployment/lz_q1l_straight_20261105/README.md) | TorchScript、配置、SDK快照、校验与只读采集工具 |
| [ROS2 消息包](deployment/ros2_ws/src/robot_common_interface/package.xml) | 缺少厂商消息环境时可单独编译的接口定义 |
| [历史项目说明](deployment/history/repository_readme_20261101.md) | 原训练环境、四个旧模型及恢复方法 |

当前直行模型：`deployment/lz_q1l_straight_20261105/policy/model700_bands.pt`，SHA-256 `109c87fb85abb565840990497a04aa003632cf164896b332073b9246545a5e30`。旧的 `models/forward/model_800.pt` 等保留作为历史模型，不是此次实机接入的默认选择。

原有 `HIMLoco/`、`tools/`、`models/`、`docs/`、`media/` 保存既有训练与仿真资料。旧脚本中的服务器绝对路径和未收录父模型需要单独处理；部署包中的历史路径仅用于追溯，运行入口不依赖它们。上游许可证见 `HIMLoco/LICENSE`，厂商接口包声明见其 `package.xml`。
