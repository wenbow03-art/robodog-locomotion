"""Evaluate the original and reflected V8 controller with paired initial states."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys

import isaacgym  # noqa: F401
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from calibrate_lz_q1l_estimator import (
    ROOT, SOURCE, BLOCK, CASES, REPEATS, DURATION, WARMUP,
    CalibrationRobot, evaluate, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--transitions', action='store_true')
    opts = parser.parse_args()
    out = opts.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError(out)
    sources = [Path(__file__).resolve(), ROOT / 'tools/calibrate_lz_q1l_estimator.py',
               ROOT / 'tools/eval_lz_q1l_switches.py',
               ROOT / 'tools/test_lz_q1l_mirror.py', ROOT / 'tools/run_lz_q1l_mirror.sh',
               ROOT / 'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py',
               ROOT / 'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
               ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
               ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
               ROOT / 'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf',
               ROOT / 'HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py',
               ROOT / 'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
               ROOT / 'HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py']
    (out / 'provenance').mkdir()
    hashes = {}
    for path in sources:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, out / 'provenance' / path.name)
    model_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = dict(checkpoint=str(SOURCE), checkpoint_sha256=model_hash,
                    source_sha256s=hashes, seeds=opts.seeds, repeats_per_seed=REPEATS,
                    duration_s=DURATION, warmup_s=WARMUP, cases=CASES,
                    policy_mode='left-mirror',
                    method='Default runner inference: mirror only pure-left commands; original policy otherwise; unchanged checkpoint.',
                    protocol='Original and mirrored policies have identical commands and paired initial root velocities. Flat ground, no observation/action noise or domain randomization. Tracking censored after first termination; all first failures counted.',
                    limitations=['Different command cases have different randomized starts.',
                                 'Robot geometry/inertia are not exactly mirror symmetric.',
                                 'Nominal simulation only; no real-robot or robustness claim.'])
    write_json(out / 'manifest.json', manifest)
    sys.argv = [sys.argv[0], '--task', 'lz_q1l_flat_v8', '--headless', '--seed', '101',
                '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
    args = get_args()
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs = BLOCK * 2
    cfg.env.episode_length_s = 17.
    cfg.commands.curriculum = cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1e6
    cfg.noise.add_noise = False
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    task_registry.task_classes[args.task] = CalibrationRobot
    env = None
    try:
        torch.manual_seed(0)
        witness = torch.randn(8, 8, device='cuda:0')
        assert torch.isfinite(witness @ witness).all()
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train_cfg), log_dir=None, device=env.device)
        original = runner.alg.actor_critic
        original.load_state_dict(torch.load(SOURCE, map_location='cpu')['model_state_dict'], strict=True)
        original.eval()
        for parameter in original.parameters():
            parameter.requires_grad_(False)
        mirror = LZQ1LMirror.from_env(env)
        candidate = runner.get_inference_policy(device=env.device)
        assert candidate.policy_mode == 'left-mirror'
        probe = torch.randn(4, env.num_obs, device=env.device)
        assert torch.equal(mirror.history(mirror.history(probe)), probe)
        manifest['dof_names'] = list(env.dof_names)
        manifest['joint_permutation'] = mirror.permutation.cpu().tolist()
        manifest['joint_sign'] = mirror.sign.cpu().tolist()
        write_json(out / 'manifest.json', manifest)
        results = []
        for seed in opts.seeds:
            result = evaluate(env, original, candidate, seed, 'mirror', out)
            for row in result['cases']:
                row['mode'] = 'original' if row['mode'] == 'baseline' else 'left-mirror'
            write_json(out / ('eval_mirror_seed%d.json' % seed), result)
            results.append(result)
        rows = []
        for mode in ('original', 'left-mirror'):
            for case, command in CASES:
                parts = [row for result in results for row in result['cases']
                         if row['mode'] == mode and row['case'] == case]
                row = dict(mode=mode, case=case, command=command,
                           trials=len(opts.seeds) * REPEATS,
                           failures=sum(part['failures'] for part in parts),
                           valid_fraction=float(np.mean([p['valid_fraction'] for p in parts])))
                valid = [part for part in parts if 'mean' in part]
                if valid:
                    weights = [part['valid_fraction'] for part in valid]
                    for metric in ('mean', 'mae', 'estimated_vy_mean', 'predicted_next_vy_mae'):
                        row[metric] = np.average([p[metric] for p in valid], axis=0, weights=weights).tolist()
                rows.append(row)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == model_hash
        write_json(out / 'summary.json', dict(checkpoint_unchanged=True, cases=rows,
                    state_columns=['vx', 'vy', 'yaw_rate', 'base_z', 'tilt_deg'],
                    mae_columns=['vx', 'vy', 'yaw_rate'], protocol=manifest['protocol']))
        print('RESULT', out / 'summary.json', flush=True)
        for row in rows:
            if row['case'] in ('left_04', 'right_04', 'left_02', 'right_02', 'forward_05'):
                print('SUMMARY', row, flush=True)
        if opts.transitions:
            from eval_lz_q1l_switches import evaluate_switches
            for seed in opts.seeds:
                evaluate_switches(env, original, candidate, seed, 'left-mirror', out)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
