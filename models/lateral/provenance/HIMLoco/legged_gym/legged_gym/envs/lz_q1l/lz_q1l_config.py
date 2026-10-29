from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class LZQ1LFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187  # base_lin_vel, external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # Kept on: the base env always samples height points; on a plane they are zeros.
        measure_heights = True
        curriculum = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.36]  # x,y,z [m]
        # All-zero URDF pose: knees bent ~90 deg, feet under the hips, base ~0.31 m.
        default_joint_angles = {
            'roll_leg_f_l_joint': 0.0,
            'roll_leg_f_r_joint': 0.0,
            'roll_leg_b_l_joint': 0.0,
            'roll_leg_b_r_joint': 0.0,

            'pitch01_leg_f_l_joint': 0.0,
            'pitch01_leg_f_r_joint': 0.0,
            'pitch01_leg_b_l_joint': 0.0,
            'pitch01_leg_b_r_joint': 0.0,

            'pitch02_leg_f_l_joint': 0.0,
            'pitch02_leg_f_r_joint': 0.0,
            'pitch02_leg_b_l_joint': 0.0,
            'pitch02_leg_b_r_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # With kp=40 the zero-action pose sags into a ~14 deg nose-up stance; 80 holds it level.
        stiffness = {'joint': 80.0}  # [N*m/rad]
        damping = {'joint': 2.0}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4
        hip_reduction = 1.0

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # [m/s]
            lin_vel_y = [-0.6, 0.6]  # [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # [rad/s]
            heading = [-3.14, 3.14]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lz_q1l/urdf/lz_q1l.urdf'
        name = "lz_q1l"
        foot_name = "pitch03"
        penalize_contacts_on = ["pitch01", "pitch02"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0

        only_positive_rewards = False
        tracking_sigma = 0.25
        soft_dof_pos_limit = 1.
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.30
        max_contact_force = 100.
        # Foot-link origin in the base frame: about -0.27 in stance, so -0.19 asks for ~8 cm swing clearance.
        clearance_height_target = -0.19


class LZQ1LFlatCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'flat_lz_q1l'
        max_iterations = 3000
        save_interval = 100


class LZQ1LFlatV2Cfg(LZQ1LFlatCfg):
    """v1 crouched (~0.24 m) at zero command and ignored pure lateral / yaw commands.

    Trained from scratch these penalties make early episodes net-negative and the policy learns
    to fall immediately, so v2 is fine-tuned from a v1 checkpoint (tools/train_lz_q1l.py --init_from).
    """

    class commands(LZQ1LFlatCfg.commands):
        zero_command_prob = 0.15

    class rewards(LZQ1LFlatCfg.rewards):
        class scales(LZQ1LFlatCfg.rewards.scales):
            termination = -50.0
            orientation = -1.0
            base_height = -10.0
            stand_still = -0.5
            feet_air_time = 0.3
            collision = -1.0
            dof_pos_limits = -2.0
            torque_limits = -0.01

        feet_air_time_target = 0.3
        soft_dof_pos_limit = 0.95
        soft_torque_limit = 0.9


class LZQ1LFlatV2CfgPPO(LZQ1LFlatCfgPPO):
    class runner(LZQ1LFlatCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v2'
        max_iterations = 1500


class LZQ1LFlatV3Cfg(LZQ1LFlatV2Cfg):
    """v2 stood at 0.31 m but still did not step in place for pure yaw / lateral commands,
    and knee speed exceeded the 20 rad/s bound. Fine-tuned from a v2 checkpoint."""

    class commands(LZQ1LFlatV2Cfg.commands):
        pure_yaw_prob = 0.15
        pure_lateral_prob = 0.10

    class rewards(LZQ1LFlatV2Cfg.rewards):
        class scales(LZQ1LFlatV2Cfg.rewards.scales):
            dof_vel_limits = -1.0

        soft_dof_vel_limit = 0.9


class LZQ1LFlatV3CfgPPO(LZQ1LFlatCfgPPO):
    class runner(LZQ1LFlatCfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v3'
        max_iterations = 1000


class LZQ1LFlatV4Cfg(LZQ1LFlatV3Cfg):
    """v3 with only the joint velocity limit penalty disabled; fine-tune from v2."""

    class rewards(LZQ1LFlatV3Cfg.rewards):
        class scales(LZQ1LFlatV3Cfg.rewards.scales):
            dof_vel_limits = 0.0


class LZQ1LFlatV4CfgPPO(LZQ1LFlatV3CfgPPO):
    class runner(LZQ1LFlatV3CfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v4'
        max_iterations = 100


class LZQ1LFlatV5Cfg(LZQ1LFlatV4Cfg):
    """v4 with 15% dedicated forward commands; fine-tune from v4 model_100."""

    class commands(LZQ1LFlatV4Cfg.commands):
        pure_forward_prob = 0.15


class LZQ1LFlatV5CfgPPO(LZQ1LFlatV4CfgPPO):
    class runner(LZQ1LFlatV4CfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v5'


class LZQ1LFlatV6Cfg(LZQ1LFlatV5Cfg):
    """v5 with 10% pure forward and 20% pure lateral commands; fine-tune from v2 model_100."""

    class commands(LZQ1LFlatV5Cfg.commands):
        pure_forward_prob = 0.10
        pure_lateral_prob = 0.20


class LZQ1LFlatV6CfgPPO(LZQ1LFlatV5CfgPPO):
    class runner(LZQ1LFlatV5CfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v6'
        max_iterations = 50
        save_interval = 50


class LZQ1LFlatV7Cfg(LZQ1LFlatV6Cfg):
    """V6 plus pure lateral tracking reward and an initial low-speed stage.

    Fine-tune from V6 model_50. Expand the dedicated lateral speed range only
    after evaluating both directions and checking forward/standing regression.
    """

    class commands(LZQ1LFlatV6Cfg.commands):
        # Absolute speeds, sampled with equal probability for either sign.
        # This range applies only to the dedicated 20% lateral command bucket.
        pure_lateral_speed_range = [0.15, 0.25]

    class rewards(LZQ1LFlatV6Cfg.rewards):
        # Denominator of exp(-vy_error**2 / sigma), not a standard deviation.
        lateral_tracking_sigma = 0.04

        class scales(LZQ1LFlatV6Cfg.rewards.scales):
            tracking_lateral = 1.5


class LZQ1LFlatV7CfgPPO(LZQ1LFlatV6CfgPPO):
    class runner(LZQ1LFlatV6CfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v7'


class LZQ1LFlatV8Cfg(LZQ1LFlatV7Cfg):
    """V7 with broader pure lateral commands and a normalized error penalty."""

    class commands(LZQ1LFlatV7Cfg.commands):
        pure_lateral_prob = 0.25
        # Absolute speeds in m/s; left and right are sampled equally.
        pure_lateral_speed_range = [0.25, 0.45]

    class rewards(LZQ1LFlatV7Cfg.rewards):
        class scales(LZQ1LFlatV7Cfg.rewards.scales):
            lateral_error = -0.5


class LZQ1LFlatV8CfgPPO(LZQ1LFlatV7CfgPPO):
    class runner(LZQ1LFlatV7CfgPPO.runner):
        experiment_name = 'flat_lz_q1l_v8'
