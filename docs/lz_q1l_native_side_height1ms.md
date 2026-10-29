# 并行直接1ms侧移训练

用户要求保留加重高度惩罚参数，同时直接训练1ms。新任务 `lz_q1l_flat_native_side_height1ms` 与运行中的4ms组并行；新组同样300轮，每50轮保存，不终止4ms组。

两组从相同side_speed model250（SHA d6457eac165260a133a9fbeaff04015198621f915dbc945c33acf3a81f4478e4）开始，精确恢复网络和两套Adam存档；估计器保持冻结且无独立更新，actor/critic/std继续优化。新阶段独立编号0～300。

保持高度目标0.30m、base_height=-100、tracking_lateral=3，其他奖励相同；左右固定各50%，速度绝对值0.25～0.45m/s，课程关闭，无老师，训练和默认回放无镜像。上次左移镜像仅是用户要求的单次回放。

唯一动力学参数差异：
- 4ms物理/decimation5 → 1ms物理/decimation20。策略周期均20ms。
- 延迟步数[0,1,2,4] → [0,4,8,16]，实际延迟均为[0,4,8,16]ms，概率不变。

预检核对与4ms组已存env_config只有这三处差异、实际延迟和动作周期一致、相同源SHA；32环境100步和一次丢弃PPO更新验证有限值、原生动作一致、actor更新、估计器不变、老师不存在、优化器恢复准确。通过不代表侧移学习成功；直接跳1ms是否能适应需训练后仿真对比。

配置追加在 `lz_q1l_2_config.py`，类 `LZQ1LNativeSideHeightOneMsCfg/PPO`。入口 `tools/launch_lz_q1l_native_side_height1ms.py`，选择除4ms训练GPU之外空闲显存最多者。状态在 `media/native_side_height1ms_20261029/latest_training.json` 指向的run/status.json。不自动部署、不按质量提前终止训练。
