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


class LZQ1LFlatV2GaitHeightCfg(LZQ1LFlatV2GaitCfg):
    """Stronger height tracking and standing-only load distribution penalties."""

    class rewards(LZQ1LFlatV2GaitCfg.rewards):
        stand_force_window_s = 0.50
        stand_force_grace_s = 0.50
        stand_pair_force_tolerance = 0.10
        stand_pair_force_epsilon_n = 1.0
        stand_min_load_fraction = 0.10

        class scales(LZQ1LFlatV2GaitCfg.rewards.scales):
            base_height = -30.0  # Triple the penalty for the same height error.
            stand_pair_force_balance = -0.50
            stand_low_load = -0.20


class LZQ1LFlatV2GaitHeightCfgPPO(LZQ1LFlatV2GaitCfgPPO):
    class runner(LZQ1LFlatV2GaitCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_gait_height'
        max_iterations = 300  # User-authorized new updates from gait model_200.


class LZQ1LFlatV2UprightCfg(LZQ1LFlatV2GaitHeightCfg):
    """Continue height/balance model_50 with stronger roll/pitch penalty."""

    class rewards(LZQ1LFlatV2GaitHeightCfg.rewards):
        class scales(LZQ1LFlatV2GaitHeightCfg.rewards.scales):
            orientation = -1.0  # Previously -0.2; applies to all commands.


class LZQ1LFlatV2UprightCfgPPO(LZQ1LFlatV2GaitHeightCfgPPO):
    class runner(LZQ1LFlatV2GaitHeightCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_upright'
        max_iterations = 150  # Additional updates, stage labels 50/100/150.


class LZQ1LFlatV2StandFrontCfg(LZQ1LFlatV2UprightCfg):
    """Standing-only posture repair and rolling front-load balance in straight walking."""

    class rewards(LZQ1LFlatV2UprightCfg.rewards):
        stand_posture_grace_s = 0.50
        front_force_window_s = 1.0
        front_force_imbalance_tolerance = 0.10
        front_force_denominator_epsilon_n = 1.0
        front_force_min_total_n = 10.0

        class scales(LZQ1LFlatV2UprightCfg.rewards.scales):
            stand_height_extra = -60.0  # Standing total: -30 -60 = -90 after grace.
            stand_orientation_extra = -1.0  # Standing total: -1 -1 = -2 after grace.
            front_force_balance = -1.0


class LZQ1LFlatV2StandFrontCfgPPO(LZQ1LFlatV2UprightCfgPPO):
    class runner(LZQ1LFlatV2UprightCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_stand_front'
        max_iterations = 300
        save_interval = 50


class LZQ1LFlatV2BalanceRollCfg(LZQ1LFlatV2UprightCfg):
    """From upright150: add only walking front balance and all-command roll cost."""

    class rewards(LZQ1LFlatV2UprightCfg.rewards):
        front_force_window_s = 1.0
        front_force_imbalance_tolerance = 0.10
        front_force_denominator_epsilon_n = 1.0
        front_force_min_total_n = 10.0

        class scales(LZQ1LFlatV2UprightCfg.rewards.scales):
            front_force_balance = -1.0
            body_roll = -1.0


class LZQ1LFlatV2BalanceRollCfgPPO(LZQ1LFlatV2UprightCfgPPO):
    class runner(LZQ1LFlatV2UprightCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_balance_roll'
        max_iterations = 200
        save_interval = 50


class LZQ1LFlatV2StandLoadCfg(LZQ1LFlatV2GaitCfg):
    """From gait150: add standing load protection, preserving gait and height."""

    class rewards(LZQ1LFlatV2GaitCfg.rewards):
        stand_force_window_s = 0.50
        stand_force_grace_s = 0.50
        stand_pair_force_tolerance = 0.10
        stand_pair_force_epsilon_n = 1.0
        stand_min_load_fraction = 0.10

        class scales(LZQ1LFlatV2GaitCfg.rewards.scales):
            stand_pair_force_balance = -0.50
            stand_low_load = -0.20


class LZQ1LFlatV2StandLoadCfgPPO(LZQ1LFlatV2GaitCfgPPO):
    class runner(LZQ1LFlatV2GaitCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_stand_load'
        max_iterations = 200  # New updates from gait model_150.
        save_interval = 10


class LZQ1LFlatV2SideLoadCfg(LZQ1LFlatV2GaitCfg):
    """Original gait150 plus a one-sided excess left-support cost."""

    class rewards(LZQ1LFlatV2GaitCfg.rewards):
        side_force_window_s = 1.0
        side_force_epsilon_n = 1.0

        class scales(LZQ1LFlatV2GaitCfg.rewards.scales):
            left_load_excess = -1.0


class LZQ1LFlatV2SideLoadCfgPPO(LZQ1LFlatV2GaitCfgPPO):
    class runner(LZQ1LFlatV2GaitCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_side_load'
        max_iterations = 100
        save_interval = 10


class LZQ1LFlatV2SideLoadStrongCfg(LZQ1LFlatV2SideLoadCfg):
    """Candidate from gait150: 10x left-only load cost, original posture weight."""

    class rewards(LZQ1LFlatV2SideLoadCfg.rewards):
        class scales(LZQ1LFlatV2SideLoadCfg.rewards.scales):
            left_load_excess = -10.0  # Was -1.0; right-heavy states still cost zero.


class LZQ1LFlatV2SideLoadStrongCfgPPO(LZQ1LFlatV2SideLoadCfgPPO):
    class runner(LZQ1LFlatV2SideLoadCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2_side_load_strong'
        max_iterations = 300  # User-authorized new updates from original gait150.


class LZQ1LFlatV3RearTeacherCfg(LZQ1LFlatV2GaitCfg):
    """New v3 keeps the gait150 environment/rewards exactly unchanged."""
    pass


class LZQ1LFlatV3RearTeacherCfgPPO(LZQ1LFlatV2GaitCfgPPO):
    class algorithm(LZQ1LFlatV2GaitCfgPPO.algorithm):
        rear_teacher_coef = 0.10
        other_legs_coef = 0.02
        teacher_max_samples = 4096
        teacher_stand_threshold = 0.05
        teacher_forward_threshold = 0.10
        teacher_other_command_tolerance = 0.05

    class runner(LZQ1LFlatV2GaitCfgPPO.runner):
        algorithm_class_name = 'HIMRearTeacherPPO'
        experiment_name = 'flat_lz_q1l_v3_rear_teacher'
        max_iterations = 0  # Review only; dedicated entrypoint requires an explicit count.
        save_interval = 10


class LZQ1LFlatV3RightRearOnlyCfg(LZQ1LFlatV2GaitCfg):
    """Review candidate: baseline environment, only RH action policy can learn."""
    pass


class LZQ1LFlatV3RightRearOnlyCfgPPO(LZQ1LFlatV3RearTeacherCfgPPO):
    class policy(LZQ1LFlatV3RearTeacherCfgPPO.policy):
        right_rear_hidden_dims = [128, 64]

    class algorithm(LZQ1LFlatV3RearTeacherCfgPPO.algorithm):
        other_legs_coef = 0.0  # Other three legs are structurally frozen.
        learning_rate = 2.25e-5  # Baseline's saved LR; fresh optimizer for new parameters.

    class runner(LZQ1LFlatV3RearTeacherCfgPPO.runner):
        policy_class_name = 'HIMRightRearOnlyActorCritic'
        algorithm_class_name = 'HIMRightRearOnlyPPO'
        experiment_name = 'flat_lz_q1l_v3_right_rear_only'
        max_iterations = 0  # Review only. User's training STOP remains in force.


class LZQ1LFlatV3RightRearSplitCfgPPO(LZQ1LFlatV3RightRearOnlyCfgPPO):
    """Continue RH-only300: standing teacher .1, forward teacher .3."""
    class algorithm(LZQ1LFlatV3RightRearOnlyCfgPPO.algorithm):
        rear_teacher_coef = 0.10
        rear_teacher_forward_coef = 0.30

    class runner(LZQ1LFlatV3RightRearOnlyCfgPPO.runner):
        algorithm_class_name = 'HIMRightRearSplitPPO'
        experiment_name = 'flat_lz_q1l_v3_right_rear_split'


class LZQ1LStandWalkCfg(LZQ1LFlatV3RightRearOnlyCfg):
    """Playback-only routing; physics and rewards remain those of gait150."""
    class inference:
        blend_time_s = 0.30
        stand_threshold = 0.05
        forward_threshold = 0.10
        other_command_tolerance = 0.05


class LZQ1LStandWalkCfgPPO(LZQ1LFlatV3RightRearSplitCfgPPO):
    class runner(LZQ1LFlatV3RightRearSplitCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_stand_walk'
        max_iterations = 0  # Inference only; this change authorizes no training.


class LZQ1LFlatV4LateralCfg(LZQ1LStandWalkCfg):
    """New baseline + isolated lateral learning. Initial low-speed review stage."""
    class commands(LZQ1LStandWalkCfg.commands):
        curriculum = False
        zero_command_prob = 0.20
        pure_forward_prob = 0.20
        pure_lateral_prob = 0.60  # Within lateral samples, left/right balanced 50/50; no mirror.
        pure_yaw_prob = 0.0
        pure_lateral_speed_range = [0.15, 0.25]

    class rewards(LZQ1LStandWalkCfg.rewards):
        class scales(LZQ1LStandWalkCfg.rewards.scales):
            tracking_lateral = 1.5  # Old V7; existing base planar/yaw tracking retained.
            lateral_error = -0.5  # Old V8 normalized absolute lateral error.
            lateral_swing_clearance = -0.20  # Same geometry as current forward gait.
            lateral_feet_slide = -0.20


class LZQ1LFlatV4LateralCfgPPO(LZQ1LFlatV2GaitCfgPPO):
    # Inherit ordinary PPO settings, NOT V3's LH-to-RH distillation settings.
    class policy(LZQ1LFlatV2GaitCfgPPO.policy):
        right_rear_hidden_dims = [128, 64]
        lateral_hidden_dims = [256, 128]
        lateral_noise_std = 0.20

    class algorithm(LZQ1LFlatV2GaitCfgPPO.algorithm):
        learning_rate = 3e-4
        schedule = 'fixed'  # Avoid inherited adaptive escalation to .01.

    class runner(LZQ1LFlatV2GaitCfgPPO.runner):
        runner_class_name = 'HIMLateralRunner'
        policy_class_name = 'HIMLateralActorCritic'
        algorithm_class_name = 'HIMLateralPPO'
        experiment_name = 'flat_lz_q1l_v4_lateral'
        baseline_directory = 'logs2/baselines/stand_walk800_20261027'
        max_iterations = 500  # User-authorized new V4 updates, direct left/right learning.
        save_interval = 50


class LZQ1LLegacyLateralCfg(LZQ1LStandWalkCfg):
    """Playback composite; all physics/control settings inherited from baseline."""
    pass


class LZQ1LLegacyLateralCfgPPO(LZQ1LStandWalkCfgPPO):
    class runner(LZQ1LStandWalkCfgPPO.runner):
        runner_class_name = 'HIMLegacyLateralRunner'
        experiment_name = 'flat_lz_q1l_baseline_v8_lateral'
        resume = False
        max_iterations = 0
        baseline_checkpoint = 'logs2/baselines/stand_walk800_20261027/model_800.pt'
        baseline_sha256 = '23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
        lateral_checkpoint = 'logs1/flat_lz_q1l_v8/v8_ft_v7_200_200iters_20261018_130426_2712609/model_200.pt'
        lateral_sha256 = '3480ca2abddfcd62df2be30d89e53a992e3eadc100add59df131213feef48a5a'


from .lz_q1l_config import LZQ1LFlatV8Cfg, LZQ1LFlatV8CfgPPO


class LZQ1LLateralCycleCfg(LZQ1LFlatV8Cfg):
    """Old V8 dynamics and commands; add only a pure-lateral stride-period cost."""
    class rewards(LZQ1LFlatV8Cfg.rewards):
        lateral_cycle_target_s = 0.80
        lateral_cycle_tolerance_s = 0.10
        lateral_cycle_min_air_s = 0.06
        lateral_cycle_contact_s = 0.04
        lateral_cycle_min_clearance_m = 0.01
        lateral_cycle_air_n = 1.0
        lateral_cycle_contact_n = 5.0

        class scales(LZQ1LFlatV8Cfg.rewards.scales):
            lateral_cycle = -0.50


class LZQ1LLateralCycleCfgPPO(LZQ1LFlatV8CfgPPO):
    class runner(LZQ1LFlatV8CfgPPO.runner):
        experiment_name = 'flat_lz_q1l_lateral_cycle'
        log_directory = 'logs2'
        max_iterations = 200
        save_interval = 50
        resume = False


class LZQ1LV8FourMsCfg(LZQ1LFlatV8Cfg):
    """First dynamics adaptation stage; original V8 rewards and commands."""
    class sim(LZQ1LFlatV8Cfg.sim):
        dt = 0.004

    class control(LZQ1LFlatV8Cfg.control):
        decimation = 5

    class domain_rand(LZQ1LFlatV8Cfg.domain_rand):
        # Four delay choices approximating original0/5/10/15ms on a4ms grid.
        action_delay_steps = [0, 1, 2, 4]


class LZQ1LV8FourMsCfgPPO(LZQ1LFlatV8CfgPPO):
    class algorithm(LZQ1LFlatV8CfgPPO.algorithm):
        learning_rate = 1e-5
        schedule = 'fixed'

    class runner(LZQ1LFlatV8CfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v8_4ms'
        log_directory = 'logs2'
        max_iterations = 200
        save_interval = 50
        resume = False


class LZQ1LNativeSideCfg(LZQ1LV8FourMsCfg):
    class commands(LZQ1LV8FourMsCfg.commands):
        curriculum = False
        zero_command_prob = 0.
        pure_forward_prob = 0.
        pure_yaw_prob = 0.
        pure_lateral_prob = 1.
        pure_lateral_speed_range = [0.25, 0.45]
        class ranges(LZQ1LV8FourMsCfg.commands.ranges):
            lin_vel_x = [0., 0.]
            lin_vel_y = [-.45, .45]
            ang_vel_yaw = [0., 0.]


class LZQ1LNativeSideCfgPPO(LZQ1LV8FourMsCfgPPO):
    # Separate settings keep the ordinary runner's initializer compatible.
    class anchor:
        coefficient = 1.0
        freeze_estimator = True

    class runner(LZQ1LV8FourMsCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_native_side_anchor'
        runner_class_name = 'HIMNativeSideRunner'


class LZQ1LNativeSideSpeedCfg(LZQ1LNativeSideCfg):
    """Native side-only adaptation with twice the lateral tracking reward."""
    class rewards(LZQ1LNativeSideCfg.rewards):
        class scales(LZQ1LNativeSideCfg.rewards.scales):
            tracking_lateral = 3.0  # Previously 1.5; same error curve.


class LZQ1LNativeSideSpeedCfgPPO(LZQ1LV8FourMsCfgPPO):
    # No anchor settings or teacher; only the state estimator remains frozen.
    class runner(LZQ1LV8FourMsCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_native_side_speed'
        runner_class_name = 'HIMNativeSideSpeedRunner'
        max_iterations = 300


class LZQ1LNativeSideHeightCfg(LZQ1LNativeSideSpeedCfg):
    """Increase height-error cost tenfold; target remains 0.30 m."""
    class rewards(LZQ1LNativeSideSpeedCfg.rewards):
        class scales(LZQ1LNativeSideSpeedCfg.rewards.scales):
            base_height = -100.0  # Previously -10.0.


class LZQ1LNativeSideHeightCfgPPO(LZQ1LNativeSideSpeedCfgPPO):
    class runner(LZQ1LNativeSideSpeedCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_native_side_height'
        max_iterations = 300


class LZQ1LNativeSideHeightOneMsCfg(LZQ1LNativeSideHeightCfg):
    """Parallel 1 ms branch, preserving rewards and the 20 ms policy period."""
    class sim(LZQ1LNativeSideHeightCfg.sim):
        dt = 0.001

    class control(LZQ1LNativeSideHeightCfg.control):
        decimation = 20

    class domain_rand(LZQ1LNativeSideHeightCfg.domain_rand):
        # Preserve the 4 ms branch's physical delays: 0/4/8/16 ms.
        action_delay_steps = [0, 4, 8, 16]


class LZQ1LNativeSideHeightOneMsCfgPPO(LZQ1LNativeSideHeightCfgPPO):
    class runner(LZQ1LNativeSideHeightCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_native_side_height1ms'


class LZQ1LSideCoordinationCfg(LZQ1LNativeSideHeightOneMsCfg):
    class rewards(LZQ1LNativeSideHeightOneMsCfg.rewards):
        side_load_window_s = 1.0
        side_min_load_share = 0.10
        side_min_total_force_n = 10.0
        side_contact_force_n = 5.0
        side_contact_duration_s = 0.04
        side_air_timeout_s = 0.30
        side_air_cost_cap = 4.0
        side_swing_min_s = 0.06
        side_swing_full_reward_s = 0.15
        side_swing_clearance_m = 0.01

        class scales(LZQ1LNativeSideHeightOneMsCfg.rewards.scales):
            lateral_load_floor = -1.0
            lateral_air_timeout = -0.5
            # feet_air_time remains +0.3; lateral behavior is replaced in env.


class LZQ1LSideCoordinationCfgPPO(LZQ1LNativeSideHeightOneMsCfgPPO):
    class runner(LZQ1LNativeSideHeightOneMsCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_side_coordination'
        runner_class_name = 'HIMSideCoordinationRunner'
        source_checkpoint = 'logs2/flat_lz_q1l_native_side_height1ms/native_side_height1ms_300_20261029_002802_2973756/model_300.pt'
        source_sha256 = 'b8984209bcb1711addf2bfda6739c7b7e0400d280e42d6638f66c8d26e56ec11'


class LZQ1LFrontPoseCfg(LZQ1LSideCoordinationCfg):
    class rewards(LZQ1LSideCoordinationCfg.rewards):
        front_pose_window_s = 1.0
        front_pose_deadband_deg = 10.0
        front_pose_scale_deg = 10.0
        class scales(LZQ1LSideCoordinationCfg.rewards.scales):
            lateral_front_pose = -0.03


class LZQ1LFrontPoseCfgPPO(LZQ1LSideCoordinationCfgPPO):
    class runner(LZQ1LSideCoordinationCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_front_pose'
        runner_class_name = 'HIMFrontPoseRunner'
        max_iterations = 300
        source_checkpoint = 'logs2/flat_lz_q1l_side_coordination/side_coordination_500_20261029_130610_3332474/model_500.pt'
        source_sha256 = 'e57308978f05ce3d6f91d61780cecb7a03bf9a216fdc36c3a065fac1dfde0b9b'


class LZQ1LSoftSlipCfg(LZQ1LFrontPoseCfg):
    class rewards(LZQ1LFrontPoseCfg.rewards):
        soft_slip_full_force_n = 5.0
        soft_slip_speed_tolerance = 0.03  # m/s
        soft_slip_speed_scale = 0.10  # m/s
        class scales(LZQ1LFrontPoseCfg.rewards.scales):
            lateral_soft_slip = -0.05


class LZQ1LSoftSlipCfgPPO(LZQ1LFrontPoseCfgPPO):
    class runner(LZQ1LFrontPoseCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_soft_slip'
        runner_class_name = 'HIMSideCoordinationRunner'
        # This runner restores iteration300 and both optimizers, then resets stage0.
        source_checkpoint = 'logs2/flat_lz_q1l_front_pose/front_pose_300_20261029_164847_3445246/model_300.pt'
        source_sha256 = '66ff4a30b5c904d94650c0a2401191f7f4307a3e0fa984ad36c5b544f6bd88c0'
