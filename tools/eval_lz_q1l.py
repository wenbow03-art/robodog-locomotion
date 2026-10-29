"""Headless evaluation of a trained LZ-Q1L policy on flat ground under fixed velocity commands.

Usage: python tools/eval_lz_q1l.py --task lz_q1l_flat [--load_run RUN] [--checkpoint N]
"""
import isaacgym  # noqa: F401  (must be imported before torch)
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

SCENARIOS = [  # (name, vx, vy, wz)
    ("stand", 0.0, 0.0, 0.0),
    ("fwd_0.5", 0.5, 0.0, 0.0),
    ("fwd_1.0", 1.0, 0.0, 0.0),
    ("back_0.5", -0.5, 0.0, 0.0),
    ("side_0.4", 0.0, 0.4, 0.0),
    ("fwd_0.5_side_0.3", 0.5, 0.3, 0.0),
    ("yaw_0.8", 0.0, 0.0, 0.8),
    ("fwd_0.6_yaw_0.5", 0.6, 0.0, 0.5),
]
NUM_ENVS = 64
DURATION_S = 10.0
WARMUP_S = 2.0


def main():
    args = get_args()
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.noise.add_noise = False
    env_cfg.commands.resampling_time = 1e6
    env_cfg.commands.heading_command = False
    for flag in ["randomize_friction", "push_robots", "disturbance", "randomize_payload_mass",
                 "randomize_com_displacement", "randomize_kp", "randomize_kd"]:
        setattr(env_cfg.domain_rand, flag, False)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    steps = int(DURATION_S / env.dt)
    warmup = int(WARMUP_S / env.dt)
    print("%-16s %6s %6s %6s | %6s %6s %6s | %6s %6s %7s %6s %6s" % (
        "scenario", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "falls", "h_mean", "tilt_mx", "tauMx", "dqMx"))
    for name, vx, vy, wz in SCENARIOS:
        env.reset()
        cmd = torch.tensor([vx, vy, wz], device=env.device)
        env.commands[:, :3] = cmd
        obs = env.get_observations()
        falls = 0
        vel_sum = torch.zeros(3, device=env.device)
        h_sum, n = 0.0, 0
        tilt_max, tau_max, dq_max = 0.0, 0.0, 0.0
        for i in range(steps):
            with torch.no_grad():
                actions = policy(obs)
            env.commands[:, :3] = cmd
            obs, _, _, dones, _, _, _ = env.step(actions)
            falls += int((env.reset_buf & ~env.time_out_buf).sum().item())
            if i >= warmup:
                vel_sum += torch.stack([env.base_lin_vel[:, 0], env.base_lin_vel[:, 1], env.base_ang_vel[:, 2]], 1).mean(0)
                h_sum += env.root_states[:, 2].mean().item()
                n += 1
                tilt = torch.rad2deg(torch.asin(env.projected_gravity[:, :2].norm(dim=1).clamp(max=1.0)))
                tilt_max = max(tilt_max, tilt.max().item())
                tau_max = max(tau_max, env.torques.abs().max().item())
                dq_max = max(dq_max, env.dof_vel.abs().max().item())
        v = (vel_sum / n).tolist()
        print("%-16s %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f | %6d %6.3f %7.1f %6.1f %6.1f" % (
            name, vx, vy, wz, v[0], v[1], v[2], falls, h_sum / n, tilt_max, tau_max, dq_max))


if __name__ == "__main__":
    main()
