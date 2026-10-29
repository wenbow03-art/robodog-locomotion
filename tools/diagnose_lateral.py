"""Measure fixed-command tracking, with per-environment first-failure censoring."""
import json
import sys
from pathlib import Path

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.envs.lz_q1l.lz_q1l import LZQ1LRobot
from legged_gym.utils import get_args, task_registry

CASES = [("stand", (0., 0., 0.)), ("forward", (.5, 0., 0.)),
         ("left_02", (0., .2, 0.)), ("right_02", (0., -.2, 0.)),
         ("left_04", (0., .4, 0.)), ("right_04", (0., -.4, 0.))]
REPEATS = 32


def fixed_commands(self, env_ids):
    commands = torch.tensor([v for _, v in CASES], device=self.device)
    self.commands[env_ids, :3] = commands[env_ids // REPEATS]
    self.commands[env_ids, 3] = 0.


def main():
    idx = sys.argv.index("--out")
    out = Path(sys.argv[idx + 1])
    del sys.argv[idx:idx + 2]
    args = get_args()
    args.headless = True
    cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    cfg.env.num_envs = len(CASES) * REPEATS
    cfg.env.episode_length_s = 15.
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.noise.add_noise = False
    for name in dir(cfg.domain_rand):
        if not name.startswith('_') and isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    LZQ1LRobot._resample_commands = fixed_commands
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    env.obs_buf.zero_()
    obs, _ = env.reset()
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    records, masks = [], []
    failures = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    duration, warmup = 10., 2.
    with torch.inference_mode():
        for step in range(round(duration / env.dt)):
            obs, _, _, dones, _, _, _ = env.step(policy(obs))
            failed = dones.bool() & ~env.time_out_buf
            failures |= failed & alive
            alive &= ~dones.bool()
            tilt = torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.)))
            records.append(torch.stack((env.base_lin_vel[:, 0], env.base_lin_vel[:, 1],
                                        env.base_ang_vel[:, 2], env.root_states[:, 2], tilt), -1).cpu().numpy())
            masks.append(alive.cpu().numpy().copy())
            if (step + 1) % 100 == 0:
                print('Evaluation steps:', step + 1, flush=True)
    data, valid = np.asarray(records), np.asarray(masks)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + '.npz', state=data, valid=valid, dt=env.dt)
    result = dict(task=args.task, run=args.load_run, checkpoint=args.checkpoint,
                  repeats=REPEATS, duration=duration, warmup=warmup, seed=args.seed,
                  protocol='Nominal flat ground; all domain randomization disabled; initial root velocities remain randomized; samples after first termination excluded.',
                  columns=['vx', 'vy', 'wz', 'height', 'tilt_deg'], cases=[])
    start = round(warmup / env.dt)
    failed_cpu = failures.cpu().numpy()
    for k, (name, command) in enumerate(CASES):
        sl = slice(k * REPEATS, (k + 1) * REPEATS)
        x, mask = data[start:, sl], valid[start:, sl]
        samples = x[mask]
        entry = dict(name=name, command=command, failures=int(failed_cpu[sl].sum()),
                     valid_fraction=float(mask.mean()),
                     mean=samples.mean(0).tolist() if len(samples) else None,
                     mae=np.abs(samples[:, :3] - np.array(command)).mean(0).tolist() if len(samples) else None,
                     tilt_p95=float(np.percentile(samples[:, 4], 95)) if len(samples) else None)
        result['cases'].append(entry)
        print(json.dumps(entry), flush=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
