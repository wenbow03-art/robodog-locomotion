# 新 baseline + 旧 V8 侧移组合控制器

任务：`lz_q1l_flat_baseline_v8_lateral`。仅推理，无训练、无权重融合。

- 站立：新 baseline 内的 gait150 actor/estimator。
- 直行：新 baseline 内的 model800 右后腿修正，保留站立/直行 0.3 秒渐变。
- 纯侧移（|vy|>0.05、vx/yaw≈0）：旧 V8 model200 及它自己的估计器；左移镜像全部六帧物理观测，输出镜像回真实关节坐标，右移直接推理。
- 所有动作统一使用新 baseline 的物理与 PD 设置：dt=0.001、decimation=20、50Hz 策略。配置完整继承 `LZQ1LStandWalkCfg`。
- 侧移进入/退出与左右反向当前直接选择策略，未额外增加时间混合。旧 baseline 的站立/直行渐变不变。物理状态和历史不因切换而清零，实际动作历史供两个策略读取。
- 非纯侧移指令走原 baseline；本次不宣称新增转弯或斜走能力。

配置：`lz_q1l_2_config.py` 中 `LZQ1LLegacyLateralCfg/PPO`。
路由：`utils/lz_q1l_legacy_lateral.py`。
加载器：`rsl_rl/runners/him_legacy_lateral_runner.py`，自动严格加载两份哈希固定权重，冻结全部参数。训练入口会明确报错。

新 baseline model800 SHA256：23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086。
旧 V8 model200 SHA256：3480ca2abddfcd62df2be30d89e53a992e3eadc100add59df131213feef48a5a。

仿真：`bash media/baseline_v8_lateral_20261027/record.sh`。
连续站立、直行0.35、左移0.30、右移0.30、停车，各8秒，额外初始站立预热2秒。无段间重置。视频包含两个视角、真实指令和镜像标注。
逐步核对非侧移动作与 baseline 相同，侧移动作与旧 V8 镜像包装器相同；验证物理设置、权重哈希和视频完整性。结果位置 `media/baseline_v8_lateral_20261027/latest_result.json`，存在并成功生成后才代表录制完成。

冻结参数不保证侧移后停车立即恢复最初承重；单次无扰动录像不等于鲁棒性验证。旧 V8 原生左移仍未被重新训练。
