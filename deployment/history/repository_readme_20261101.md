# LZ-Q1L机器狗训练项目

本仓库保存远程开发项目的代码与四个指定模型。包括HIMLoco环境、PPO/HIM实现、
机器人URDF与网格、训练与检查脚本，以及实验说明。上游许可证见`HIMLoco/LICENSE`。

## 保留模型

| 用途 | 文件 | 状态 |
|---|---|---|
| 站立 | `models/standing/gait_150.pt` | 用户选定baseline |
| 直行 | `models/forward/model_800.pt` | 用户选定baseline |
| 2 m/s 高速直行 | [model_500.pt](models/forward_2mps/model_500.pt) | 从 gait_150 站立平滑切换 0.3 秒；仍有横向偏移 |
| 侧移 | `models/lateral/model_300.pt` | 最新滑动优化实验版本，左移协调性仍待改善 |

每个目录包含对应训练配置、运行元数据和当时的代码快照。权重原样备份，
校验值及原路径见`models/manifest.json`。不含其他模型权重或全部训练日志。

## 代码位置

- 新系列环境与配置：`HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py`、`lz_q1l_2_config.py`。
- 原系列配置：同目录`lz_q1l_config.py`，保留用于追踪历史。
- PPO/HIM与训练器：`HIMLoco/rsl_rl/rsl_rl/`。
- 机器人：`HIMLoco/legged_gym/resources/robots/lz_q1l/`。
- 训练、仿真和检查脚本：`tools/`；选定录像脚本与结果元数据：`media/`。
- 修改说明：`docs/`。`AGENTS.md`保留历史工作记录，旧的“正在训练”等状态不是当前进度。

## 环境与恢复

原环境为Linux、Python3.8、PyTorch1.10.0+cu113、NVIDIA GPU与Isaac Gym Preview4。
依赖列表见`HIMLoco/requirements.txt`。Isaac Gym须自行取得并安装；本仓库不包含
虚拟环境、CUDA缓存、Isaac Gym安装包或整套比赛资料。

在已准备好兼容环境的机器上，从仓库根目录执行：

```bash
python tools/restore_selected_models.py
```

这仅校验并恢复四个模型到原项目相对目录，不安装环境、不训练、不覆盖不同模型。
随后按上游README安装`HIMLoco/rsl_rl`和`HIMLoco/legged_gym`，确认Python导入本仓库副本。

当前站立、直行和侧移采用1ms物理步长、decimation20，即50Hz策略输出。
统一时间设置不代表三策略任意切换已经验证，也不代表最新侧移已达到最终效果。

## 运行限制

这是代码和模型备份，不是一键可运行的迁移版本。历史脚本包含原服务器绝对路径、
已省略的其他检查点和实验元数据依赖，运行前需要改成当前目录并检查输入文件。
部分runner默认加载其父模型进行续训；这些父模型不一定在本次四个模型范围内。
回放应明确加载所选检查点，不能直接运行任意旧启动脚本。原始元数据中的路径保留
为来源记录。最新侧移快照证明训练完成，不证明所有速度、地形或真机运动可靠。
