"""Configurations for the new LZ-Q1L training lineage (logs2).

New V1 is the standing-support baseline below. Append subsequent versions and
their PPO configurations here using class inheritance, as in lz_q1l_config.py.
Keep archived V1--V8 configurations in the original lz_q1l_config.py.
"""

from .lz_q1l_config import LZQ1LFlatCfg, LZQ1LFlatCfgPPO


class LZQ1LFlatV1OneMsCfg(LZQ1LFlatCfg):
    """Shared 1 ms baseline; also used by the standalone timestep task."""

    class sim(LZQ1LFlatCfg.sim):
        dt = 0.001  # 1 ms physics / PD update, originally 5 ms.

    class control(LZQ1LFlatCfg.control):
        decimation = 20  # 20 * 1 ms = original 20 ms policy interval.

    class domain_rand(LZQ1LFlatCfg.domain_rand):
        # Preserve V1's uniform 0/5/10/15 ms delay, in new physics steps.
        action_delay_steps = [0, 5, 10, 15]


class LZQ1LFlatV1OneMsCfgPPO(LZQ1LFlatCfgPPO):
    class runner(LZQ1LFlatCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v1_1ms'


class LZQ1LFlatV1SupportCfg(LZQ1LFlatV1OneMsCfg):
    class commands(LZQ1LFlatV1OneMsCfg.commands):
        zero_command_prob = 0.20
        pure_forward_prob = 0.30
        pure_forward_speed_range = [0.20, 0.50]

    class rewards(LZQ1LFlatV1OneMsCfg.rewards):
        # These are initial experimental values, not tuned training results.
        stand_command_threshold = 0.05  # each of vx, vy, yaw must be near zero
        stand_support_grace_s = 0.50
        stand_support_air_time_s = 0.20
        stand_support_height_m = 0.005
        stand_support_contact_force_n = 1.0

        class scales(LZQ1LFlatV1OneMsCfg.rewards.scales):
            # Per offending foot, per second; base reward code multiplies by dt.
            stand_unsupported = -1.0


class LZQ1LFlatV1SupportCfgPPO(LZQ1LFlatV1OneMsCfgPPO):
    class runner(LZQ1LFlatV1OneMsCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v1_support'
        log_directory = 'logs2'
        resume = False
        max_iterations = 500
        save_interval = 50


class LZQ1LFlatV2UnifiedCfg(LZQ1LFlatV1SupportCfg):
    """Six dedicated commands; retain new V1's physics and standing support."""

    class commands(LZQ1LFlatV1SupportCfg.commands):
        curriculum = False
        pure_lateral_prob = 0.25  # Split equally between left and right.
        pure_yaw_prob = 0.25
        pure_lateral_speed_range = [0.15, 0.25]
        pure_yaw_speed_range = [0.30, 0.50]

        class ranges(LZQ1LFlatV1SupportCfg.commands.ranges):
            lin_vel_x = [0.0, 0.50]
            lin_vel_y = [-0.25, 0.25]
            ang_vel_yaw = [-0.50, 0.50]

    class rewards(LZQ1LFlatV1SupportCfg.rewards):
        lateral_tracking_sigma = 0.04  # exp(-vy_error**2 / sigma).

        class scales(LZQ1LFlatV1SupportCfg.rewards.scales):
            tracking_lateral = 1.5


class LZQ1LFlatV2UnifiedCfgPPO(LZQ1LFlatV1SupportCfgPPO):
    class runner(LZQ1LFlatV1SupportCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2'
        max_iterations = 800  # User-approved new V2 updates from new V1 model_500.
        # Dedicated launcher requires an explicit update count and initializes
        # network weights from new V1 model_500, with fresh optimizers.


class LZQ1LFlatV2PostureCfg(LZQ1LFlatV2UnifiedCfg):
    """Posture repair from new V1: only standing/forward, two reward changes."""

    class commands(LZQ1LFlatV2UnifiedCfg.commands):
        zero_command_prob = 0.50
        pure_forward_prob = 0.50
        pure_lateral_prob = 0.0
        pure_yaw_prob = 0.0

    class rewards(LZQ1LFlatV2UnifiedCfg.rewards):
        soft_dof_pos_limit = 0.95

        class scales(LZQ1LFlatV2UnifiedCfg.rewards.scales):
            base_height = -10.0
            dof_pos_limits = -2.0


class LZQ1LFlatV2PostureCfgPPO(LZQ1LFlatV2UnifiedCfgPPO):
    class runner(LZQ1LFlatV2UnifiedCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_posture'
        max_iterations = 500  # Continue the posture run from update 20 to total 500.
        save_interval = 10


class LZQ1LFlatV2GaitCfg(LZQ1LFlatV2PostureCfg):
    """Straight-walking gait repair; preserve the archived posture task."""

    class rewards(LZQ1LFlatV2PostureCfg.rewards):
        gait_min_forward_command = 0.10
        gait_other_command_tolerance = 0.05
        gait_transition_grace_s = 0.30
        swing_clearance_target_m = 0.010
        swing_unloaded_force_n = 10.0
        swing_speed_scale_m_s = 0.35
        slide_contact_force_n = 1.0
        slide_speed_deadband_m_s = 0.03
        slide_speed_scale_m_s = 0.20
        slide_normalized_speed_cap = 2.0
        rear_force_window_s = 1.0
        rear_force_imbalance_tolerance = 0.10
        rear_force_denominator_epsilon_n = 1.0
        rear_force_min_total_n = 10.0  # Avoid treating two unloaded rear feet as balanced.

        class scales(LZQ1LFlatV2PostureCfg.rewards.scales):
            foot_clearance = 0.0  # Replaced by the geometric, normalized swing term.
            swing_clearance = -0.20
            feet_slide = -0.20
            rear_force_balance = -1.0


class LZQ1LFlatV2GaitCfgPPO(LZQ1LFlatV2PostureCfgPPO):
    class runner(LZQ1LFlatV2PostureCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_gait'
        max_iterations = 200  # New gait updates after posture model_500.
        save_interval = 50
