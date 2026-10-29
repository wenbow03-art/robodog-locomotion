"""Compare an arbitrary checkpoint to V8 using the established paired protocol."""
import argparse
from pathlib import Path
import sys

import isaacgym  # noqa: F401
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from calibrate_lz_q1l_estimator import (
    SOURCE, BLOCK, CalibrationRobot, evaluate, gate, make_actor, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    opts = parser.parse_args()
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    if (opts.out_dir / 'assessment.json').exists():
        raise FileExistsError(opts.out_dir)
    sys.argv = [sys.argv[0], '--task', 'lz_q1l_flat_v8', '--headless', '--seed', '101',
                '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
    args = get_args()
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs = BLOCK * 2
    cfg.env.episode_length_s = 17.
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.noise.add_noise = False
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    task_registry.task_classes[args.task] = CalibrationRobot
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        original = make_actor(env, train_cfg, torch.load(SOURCE, map_location='cpu')['model_state_dict'])
        candidate = make_actor(env, train_cfg, torch.load(opts.checkpoint, map_location='cpu')['model_state_dict'])
        results = [evaluate(env, original, candidate, seed, 'checkpoint', opts.out_dir) for seed in opts.seeds]
        write_json(opts.out_dir / 'assessment.json', dict(checkpoint=str(opts.checkpoint),
                    seeds=opts.seeds, protocol='12 s; 2 s warmup excluded from tracking; all first failures counted; 16 paired initial root velocities per case/seed; nominal flat ground; deterministic actions',
                    **gate(results)))
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
