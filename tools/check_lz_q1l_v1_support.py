"""Behavior checks and CPU reward calibration; does not train or change weights."""
import argparse
import copy
import faulthandler
import hashlib
import json
from pathlib import Path
import sys

faulthandler.enable()
import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV1SupportRobot
from legged_gym.envs.lz_q1l.lz_q1l_2_config import LZQ1LFlatV1SupportCfg
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner

ROOT = Path(__file__).resolve().parents[1]
STARTS = np.concatenate([np.zeros((1, 6))] + [np.random.RandomState(s).uniform(-.05, .05, (4, 6)) for s in (1, 2)])
B = len(STARTS)
MODES = ['default_pd', 'v1_500_stand', 'v1_500_forward', 'v1_1700_stand']


def check_support_rules():
    robot = object.__new__(LZQ1LV1SupportRobot)
    robot.cfg = LZQ1LFlatV1SupportCfg()
    robot.num_envs, robot.num_bodies, robot.dt = 1, 4, .02
    robot.feet_indices = torch.arange(4)
    robot.commands = torch.zeros(1, 4)
    robot.support_grace_steps, robot.support_air_steps = 25, 10
    robot.support_stand_steps = torch.zeros(1, dtype=torch.long)
    robot.support_offground_steps = torch.zeros(1, 4, dtype=torch.long)
    robot.support_last_contacts = torch.zeros(1, 4, dtype=torch.bool)
    robot.support_penalized_feet = torch.zeros(1, 4, dtype=torch.bool)
    robot.support_sphere_offsets = torch.zeros(1, 4, 3)
    robot.support_sphere_radii = torch.full((4,), .02)
    robot.rigid_body_states = torch.zeros(1, 4, 13)
    robot.rigid_body_states[:, :, 6] = 1.
    robot.contact_forces = torch.zeros(1, 4, 3)
    ids = torch.tensor([0])

    def reset():
        robot._clear_support(ids)
        robot.commands.zero_()
        robot.rigid_body_states[:, :, 2] = .02
        robot.contact_forces[:, :, 2] = 30.

    def tick(n):
        for _ in range(n):
            robot._update_standing_support()

    for foot in range(4):
        reset()
        robot.rigid_body_states[0, foot, 2] = .04
        robot.contact_forces[0, foot, 2] = 0.
        tick(35)
        assert robot._reward_stand_unsupported().item() == 0., 'Grace/air threshold triggered early'
        tick(1)
        assert robot.support_penalized_feet[0, foot] and robot._reward_stand_unsupported().item() == 1.
        previous = robot.support_offground_steps.clone()
        for _ in range(3):
            assert robot._reward_stand_unsupported().item() == 1.
        assert torch.equal(previous, robot.support_offground_steps), 'Reward must not advance counters'
        robot.contact_forces[0, foot, 2] = 30.
        tick(1)
        assert robot._reward_stand_unsupported().item() == 0. and robot.support_offground_steps.sum() == 0
        robot.contact_forces[0, foot, 2] = 0.
        tick(1)
        assert robot.support_offground_steps.sum() == 0, 'Single-frame contact dropout must be filtered'
        tick(11)
        assert robot._reward_stand_unsupported().item() == 1.
        robot._clear_support(ids)
        assert robot._reward_stand_unsupported().item() == 0. and robot.support_offground_steps.sum() == 0

    for command in ((.3, 0., 0.), (0., .3, 0.), (0., 0., .3)):
        reset()
        robot.rigid_body_states[:, :, 2] = .04
        robot.contact_forces.zero_()
        robot.commands[0, :3] = torch.tensor(command)
        tick(60)
        assert robot._reward_stand_unsupported().item() == 0., 'Moving/yaw command incorrectly penalized'
        robot.commands.zero_()
        tick(35)
        assert robot._reward_stand_unsupported().item() == 0.
        tick(1)
        assert robot._reward_stand_unsupported().item() == 4.
    reset()
    robot.contact_forces.zero_()
    robot.rigid_body_states[:, :, 2] = .024
    tick(60)
    assert robot._reward_stand_unsupported().item() == 0., 'Below-height threshold penalized'
    # With a rotated sphere offset, use world-space sphere bottom, not link height.
    reset()
    robot.rigid_body_states[:, :, 2] = .04
    robot.rigid_body_states[:, :, 3:7] = torch.tensor([0., 1., 0., 0.])
    robot.support_sphere_offsets[:, :, 2] = .02
    robot.contact_forces.zero_()
    tick(60)
    assert torch.allclose(robot.support_foot_gap, torch.zeros(1, 4), atol=1e-7)
    assert robot._reward_stand_unsupported().item() == 0.
    return 'passed: all four feet, timing, landing, contact filter, reset, all command axes, height geometry'


class ProbeRobot(LZQ1LV1SupportRobot):
    def _reset_root_states(self, ids):
        self.root_states[ids] = self.base_init_state
        self.root_states[ids, :3] += self.env_origins[ids]
        self.root_states[ids, 7:13] = torch.as_tensor(STARTS, dtype=torch.float, device=self.device)[ids % B]
        indices = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(indices), len(indices))

    def compute_reward(self):
        super().compute_reward()
        self.probe_support_reward = (self._reward_stand_unsupported() * self.reward_scales['stand_unsupported']).clone()
        self.probe_bad_feet = self.support_penalized_feet.clone()


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--out', type=Path, required=True)
    opts, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    out = opts.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg, train = task_registry.get_cfgs('lz_q1l_flat_v1_support')
    default_config = class_to_dict(cfg)
    cfg.env.num_envs = args.num_envs = B * len(MODES)
    cfg.env.episode_length_s = 20.
    cfg.commands.curriculum = False
    cfg.commands.resampling_time = 1e6
    cfg.noise.add_noise = False
    cfg.sim.use_gpu_pipeline = cfg.sim.physx.use_gpu = False
    cfg.sim.physx.num_threads = 2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    task_registry.task_classes['lz_q1l_flat_v1_support'] = ProbeRobot
    env = None
    try:
        env, _ = task_registry.make_env('lz_q1l_flat_v1_support', args=args, env_cfg=cfg)
        # BaseTask changes legacy PyTorch JIT executor flags. Initialize it before
        # exercising scripted quaternion helpers, so their cached plans stay valid.
        rules = check_support_rules()
        assert abs(env.dt - .02) < 1e-8 and env.cfg.control.decimation == 20
        counts = torch.zeros(3, dtype=torch.long, device=env.device)
        ids = torch.arange(env.num_envs, device=env.device)
        for _ in range(1000):
            env._resample_commands(ids)
            modes = env.command_sample_mode
            counts += torch.bincount(modes, minlength=3)
            assert (env.commands[modes == 0] == 0).all()
            forward = env.commands[modes == 1]
            assert (forward[:, 0] >= .2).all() and (forward[:, 0] <= .5).all()
            assert (forward[:, 1:] == 0).all()
        proportions = (counts.float() / counts.sum()).cpu().tolist()
        assert max(abs(a - b) for a, b in zip(proportions, [.2, .3, .5])) < .02
        original_range = env.command_ranges['lin_vel_x'][:]
        env.command_ranges['lin_vel_x'] = [-.35, .35]
        for _ in range(20):
            env._resample_commands(ids)
            assert (env.commands[:, 0].abs() <= .35).all(), 'Configured vx range ignored'
        env.command_ranges['lin_vel_x'] = original_range
        runner = HIMOnPolicyRunner(env, class_to_dict(train), log_dir=None, device=env.device)
        actors, checkpoints = {}, []
        for iteration in (500, 1700):
            path = ROOT / ('HIMLoco/legged_gym/logs1/flat_lz_q1l/Oct04_19-37-42_v1_kp80/model_%d.pt' % iteration)
            actor = copy.deepcopy(runner.alg.actor_critic)
            actor.load_state_dict(torch.load(path, map_location=env.device)['model_state_dict'])
            actor.eval()
            actors[iteration] = actor
            checkpoints.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        # Intentional raw V1 checkpoint baseline for reward calibration, not a demonstration.
        obs, _ = env.reset()
        env._clear_support(ids)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        data = {key: [] for key in ('valid', 'gap', 'bad_feet', 'support_reward', 'total_reward', 'vx', 'force')}
        resets = []
        with torch.no_grad():
            for step in range(600):
                env.commands[:] = 0.
                if step >= 100:
                    env.commands[2 * B:3 * B, 0] = .5
                obs[:, :3] = env.commands[:, :3] * env.commands_scale
                actions = torch.zeros(env.num_envs, 12, device=env.device)
                if step >= 100:
                    actions[B:3 * B] = actors[500].act_inference(obs[B:3 * B])
                    actions[3 * B:] = actors[1700].act_inference(obs[3 * B:])
                obs, _, reward, done, _, _, _ = env.step(actions)
                for index in (alive & done.bool()).nonzero().flatten().tolist():
                    resets.append(dict(index=index, time_s=(step + 1) * env.dt))
                alive &= ~done.bool()
                values = dict(valid=alive, gap=env.support_foot_gap, bad_feet=env.probe_bad_feet,
                    support_reward=env.probe_support_reward, total_reward=reward,
                    vx=env.base_lin_vel[:, 0], force=env.contact_forces[:, env.feet_indices, 2])
                for key, value in values.items():
                    data[key].append(value.cpu().numpy().copy())
                if (step + 1) % 100 == 0:
                    print('STEP', step + 1, flush=True)
        arrays = {key: np.asarray(value) for key, value in data.items()}
        assert all(np.isfinite(value).all() for value in arrays.values())
        rows = []
        for mi, mode in enumerate(MODES):
            ids_np = list(range(mi * B, (mi + 1) * B))
            valid = arrays['valid'][200:, ids_np]
            assert valid.all(), (mode, 'unexpected termination during calibration')
            row = dict(mode=mode, trials=B,
                penalized_foot_rate=arrays['bad_feet'][200:, ids_np][valid].mean(0).tolist(),
                mean_gap_mm=(arrays['gap'][200:, ids_np][valid].mean(0) * 1000).tolist(),
                support_reward_per_second=float(arrays['support_reward'][200:, ids_np][valid].mean() / env.dt),
                total_reward_per_second=float(arrays['total_reward'][200:, ids_np][valid].mean() / env.dt),
                mean_vx=float(arrays['vx'][200:, ids_np][valid].mean()))
            rows.append(row)
        assert rows[0]['support_reward_per_second'] == 0.
        assert rows[2]['support_reward_per_second'] == 0., 'Forward incorrectly penalized'
        assert rows[1]['penalized_foot_rate'][1] > .95 and rows[3]['penalized_foot_rate'][1] > .95
        # Actual environment reset must clear every support-state buffer for selected rows.
        env.reset_idx(ids[:3])
        assert not env.support_penalized_feet[:3].any() and not env.support_offground_steps[:3].any()
        assert not env.support_stand_steps[:3].any() and not env.support_last_contacts[:3].any()
        for checkpoint in checkpoints:
            assert hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest() == checkpoint['sha256']
        sources = [Path(__file__).resolve(), ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
            ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py']
        report = dict(status='checks_passed_no_training', rules=rules, command_proportions=proportions,
            command_draws=int(counts.sum()), default_config=default_config, diagnostic_config=class_to_dict(cfg),
            backend='CPU PhysX', modes=MODES, starts=STARTS.tolist(), resets=resets, cases=rows,
            checkpoints=checkpoints, source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
            protocol='2s default PD, then fixed raw original V1 policies; summary 4-12s. Domain randomization disabled for calibration.',
            limitations='Verifies implementation and reward magnitudes, not learned improvement or GPU training stability.')
        np.savez_compressed(out / 'traces.npz', **arrays)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(rules=rules, command_proportions=proportions, cases=rows)), flush=True)
        print('RESULT', out, flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
