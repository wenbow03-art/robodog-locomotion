"""Render a trained LZ-Q1L policy off-screen (no display needed) and save a GIF.

Usage: python tools/record_lz_q1l.py --task lz_q1l_flat [--load_run RUN] [--checkpoint N] [--out FILE]
Command schedule: stand 2 s -> forward 0.8 m/s 3 s -> forward 0.5 m/s + yaw 0.8 rad/s 3 s -> stand 2 s.
"""
import os
import sys

import isaacgym  # noqa: F401  (must be imported before torch)
from isaacgym import gymapi
import numpy as np
import torch
import imageio

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args, task_registry

SCHEDULE = [(2.0, (0.0, 0.0, 0.0)), (3.0, (0.8, 0.0, 0.0)), (3.0, (0.5, 0.0, 0.8)), (2.0, (0.0, 0.0, 0.0))]
WIDTH, HEIGHT = 640, 400

_orig_create_sim = LeggedRobot.create_sim


def _create_sim_with_graphics(self):
    self.graphics_device_id = self.sim_device_id
    _orig_create_sim(self)


LeggedRobot.create_sim = _create_sim_with_graphics


def main():
    out = "lz_q1l_walk.gif"
    if "--out" in sys.argv:
        i = sys.argv.index("--out")
        out = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    args = get_args()
    args.headless = True
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
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

    cam_props = gymapi.CameraProperties()
    cam_props.width, cam_props.height = WIDTH, HEIGHT
    cam = env.gym.create_camera_sensor(env.envs[0], cam_props)

    env.reset()
    obs = env.get_observations()
    frames = []
    step = 0
    for duration, cmd in SCHEDULE:
        cmd_t = torch.tensor(cmd, device=env.device)
        for _ in range(int(duration / env.dt)):
            env.commands[:, :3] = cmd_t
            obs[:, :3] = env.commands[:, :3] * env.commands_scale
            with torch.no_grad():
                obs, _, _, _, _, _, _ = env.step(policy(obs))
            if step % 2 == 0:
                p = env.root_states[0, :3].cpu().numpy()
                env.gym.set_camera_location(cam, env.envs[0],
                                            gymapi.Vec3(p[0] + 0.6, p[1] - 1.4, 0.55),
                                            gymapi.Vec3(p[0], p[1], 0.22))
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
                frames.append(img.reshape(HEIGHT, WIDTH, 4)[:, :, :3].copy())
            step += 1
    imageio.mimsave(out, frames, duration=2 * env.dt)
    print("saved %d frames to %s" % (len(frames), os.path.abspath(out)))


if __name__ == "__main__":
    main()
