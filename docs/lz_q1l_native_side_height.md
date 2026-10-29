# 侧移高度恢复：250轮续训300轮

用户要求增加高度惩罚并继续训练，随后明确再训练300轮。当前旧side_speed任务保存model250后结束，保留该检查点，按用户要求切换奖励。不是按质量自动提前停止。

- 新任务 `lz_q1l_flat_native_side_height`。新配置在 `lz_q1l_2_config.py` 追加 `LZQ1LNativeSideHeightCfg/PPO`。
- 唯一环境变化：`base_height` 权重-10 → -100；目标保持0.30m，公式 `-100*(measured_height-.30)^2`，实际每步乘0.02s。低于或高于目标都扣分。高度17.5cm时该项每秒尺度由-0.15625变为-1.5625。
- 侧移跟踪3.0、4ms/decimation5、左右固定各50%、速度绝对值0.25～0.45m/s、课程关闭、无老师、无镜像均保持。
- 冻结估计器、不调用其update；actor/critic/std继续PPO。完整恢复side_speed model250权重、PPO Adam和估计器Adam存档（后者保持冻结不用），固定学习率1e-5。重置仿真/RNG和阶段轮次。
- 新阶段300轮，保存model0/50/100/150/200/250/300，与源阶段250独立编号。无自动部署或下一步长阶段，无质量提前停止。
- 选择入口 `media/native_side_height_20261029/selected.json` 记录源SHA和旧进程切换状态。启动器 `tools/launch_lz_q1l_native_side_height.py`，训练脚本 `tools/train_lz_q1l_native_side_height.py`。
- 进度以 `media/native_side_height_20261029/latest_training.json` 指向的run/status.json为准；未出现该文件不表示已启动。
- 预检要求：相比源env_config仅高度scale变化；全权重和两套优化器状态精确恢复；高度30/25/17.5cm的平方误差符合公式；32环境100步和一次丢弃更新有限、actor更新且估计器全部参数不变、无老师和左右各半。

这是加重高度成本的单项修改，不保证恢复高度或步态，需要训练后实测。上一段镜像视频是回放特例，不改变本阶段训练或任务默认原生回放。
