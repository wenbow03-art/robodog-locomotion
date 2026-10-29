"""Paired zero-action PD versus V8 zero-command standing; no training changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l import LZQ1LRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner

ROOT = Path('/home/xiaohengli/Program/robo_competition')
MODEL = ROOT / 'HIMLoco/legged_gym/logs1/flat_lz_q1l_v8/v8_ft_v7_200_200iters_20261018_130426_2712609/model_200.pt'
URDF = ROOT / 'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf'
# One exact rest start and 16 small-velocity perturbations for each of two seeds.
STARTS = np.concatenate([np.zeros((1, 6))] + [
    np.random.RandomState(seed).uniform(-.05, .05, (16, 6)) for seed in (1, 2)])
BLOCK = len(STARTS)


class DiagnosticRobot(LZQ1LRobot):
    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        if hasattr(self, 'substep_probe'):
            ids = [0, BLOCK]
            values = torch.stack((self.dof_pos[ids], self.dof_vel[ids],
                                  self.joint_pos_target[ids], torques[ids]), dim=1)
            self.substep_probe.append(values.detach().cpu().numpy().copy())
        return torques

    def _resample_commands(self, ids):
        self.commands[ids] = 0.

    def _reset_root_states(self, ids):
        self.root_states[ids] = self.base_init_state
        self.root_states[ids, :3] += self.env_origins[ids]
        starts = torch.as_tensor(STARTS, dtype=torch.float, device=self.device)
        self.root_states[ids, 7:13] = starts[ids % BLOCK]
        ids32 = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--physics-dt', type=float, default=.005)
    opts, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    out = opts.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'provenance').mkdir()
    source_paths = [Path(__file__).resolve(), URDF,
        ROOT / 'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
        ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
        ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
        ROOT / 'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    for p in source_paths:
        shutil.copyfile(p, out / 'provenance' / p.name)
    model_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    cfg, train = task_registry.get_cfgs(args.task)
    cfg.sim.dt = opts.physics_dt
    cfg.control.decimation = round(.02 / opts.physics_dt)
    assert abs(cfg.sim.dt * cfg.control.decimation - .02) < 1e-9
    args.num_envs = cfg.env.num_envs = BLOCK * 2
    args.headless = True
    cfg.env.episode_length_s = 20.
    cfg.commands.curriculum = cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1e6
    cfg.noise.add_noise = False
    for name in dir(cfg.domain_rand):
        if not name.startswith('_') and isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    task_registry.task_classes[args.task] = DiagnosticRobot
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train), log_dir=None, device=env.device)
        runner.alg.actor_critic.load_state_dict(torch.load(MODEL, map_location='cpu')['model_state_dict'])
        policy = runner.get_inference_policy(device=env.device)
        bodies = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        feet = [bodies[i] for i in env.feet_indices.tolist()]
        offsets, radii = [], []
        tree = ET.parse(URDF).getroot()
        for name in feet:
            collision = tree.find("link[@name='%s']/collision" % name)
            offsets.append([float(x) for x in collision.find('origin').get('xyz').split()])
            radii.append(float(collision.find('geometry/sphere').get('radius')))
        offsets = torch.tensor(offsets, device=env.device).expand(env.num_envs, -1, -1)
        radii = torch.tensor(radii, device=env.device)
        props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
        manifest = dict(checkpoint=str(MODEL), checkpoint_sha256=model_hash,
            gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            source_sha256s=hashes, modes=['zero_action_pd', 'v8_zero_command'],
            starts=STARTS.tolist(), feet=feet, dof_names=env.dof_names,
            duration_s=12., statistics_window_s=[2., 12.], dt=env.dt,
            physics_dt=cfg.sim.dt, control_decimation=cfg.control.decimation,
            total_mass_kg=sum(p.mass for p in props),
            protocol='Paired starts: identical zero joint pose and base pose; nominal rest plus 32 small initial velocity perturbations. Plane, no noise/randomization. One shared initial zero-action step. Raw Fz>1N contact. Samples at/after first reset excluded. Sphere-bottom clearance is geometric, not contact-offset corrected. Torques sampled at last physics substep only.')
        save(out / 'manifest.json', manifest)
        save(out / 'env_config.json', class_to_dict(cfg))
        obs, _ = env.reset()
        assert not policy.mirror_mask(obs).any()
        env.substep_probe = []
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        resets = []
        history = {key: [] for key in ['valid', 'fz', 'clearance', 'q', 'target', 'torque', 'state']}
        with torch.no_grad():
            for step in range(round(12. / env.dt)):
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
                actions[BLOCK:] = policy(obs[BLOCK:])
                obs, _, _, done, _, _, _ = env.step(actions)
                for index in (alive & done.bool()).nonzero().flatten().tolist():
                    resets.append(dict(index=index, time_s=(step+1)*env.dt))
                alive &= ~done.bool()
                body = env.rigid_body_states.view(env.num_envs, env.num_bodies, 13)[:, env.feet_indices]
                rotated = quat_apply(body[:, :, 3:7].reshape(-1, 4), offsets.reshape(-1, 3)).reshape(env.num_envs, 4, 3)
                clearance = body[:, :, 2] + rotated[:, :, 2] - radii
                tilt = torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.)))
                values = dict(valid=alive, fz=env.contact_forces[:, env.feet_indices, 2],
                    clearance=clearance, q=env.dof_pos, target=env.joint_pos_target,
                    torque=env.torques, state=torch.cat((env.base_lin_vel, env.base_ang_vel,
                        env.root_states[:, 2:3], tilt[:, None]), dim=1))
                for key, value in values.items():
                    history[key].append(value.cpu().numpy().copy())
                if (step+1) % 100 == 0:
                    print('STEPS', step+1, flush=True)
        data = {key: np.asarray(value) for key, value in history.items()}
        probe = np.asarray(env.substep_probe)
        np.savez_compressed(out / 'physics_substeps.npz', data=probe,
            columns=['q', 'dq', 'target', 'torque'], dt=env.sim_params.dt)
        np.savez_compressed(out / 'traces.npz', **data)
        summaries = []
        start = round(2. / env.dt)
        for mode_i, mode in enumerate(manifest['modes']):
            for group, indices in [('nominal', [mode_i*BLOCK]),
                                   ('perturbed', list(range(mode_i*BLOCK+1, (mode_i+1)*BLOCK)))]:
                mask = data['valid'][start:, indices]
                def samples(key):
                    return data[key][start:, indices][mask]
                fz, gap, q, target = [samples(k) for k in ['fz', 'clearance', 'q', 'target']]
                assert len(fz) > 0
                row = dict(mode=mode, group=group, trials=len(indices),
                    resets=sum(r['index'] in indices for r in resets), valid_fraction=float(mask.mean()),
                    contact_fraction=(fz > 1.).mean(0).tolist(), mean_fz_N=fz.mean(0).tolist(),
                    mean_clearance_mm=(gap.mean(0)*1000).tolist(),
                    clearance_p05_mm=(np.percentile(gap, 5, axis=0)*1000).tolist(),
                    clearance_p95_mm=(np.percentile(gap, 95, axis=0)*1000).tolist(),
                    airborne_over_5mm_fraction=(gap > .005).mean(0).tolist(),
                    mean_q_rad=q.mean(0).tolist(), mean_target_rad=target.mean(0).tolist(),
                    joint_tracking_mae_rad=np.abs(q-target).mean(0).tolist(),
                    mean_torque_Nm=samples('torque').mean(0).tolist(),
                    mean_state=samples('state').mean(0).tolist(),
                    all_feet_contact_fraction=float((fz > 1.).all(-1).mean()))
                summaries.append(row)
                print('SUMMARY', json.dumps(row), flush=True)
        assert hashlib.sha256(MODEL.read_bytes()).hexdigest() == model_hash
        probe_stats = []
        for i, mode in enumerate(manifest['modes']):
            p = probe[round(2. / env.sim_params.dt):, i]
            limits = env.torque_limits.cpu().numpy()
            probe_stats.append(dict(mode=mode,
                mean_abs_dq_rad_s=np.abs(p[:, 1]).mean(0).tolist(),
                peak_abs_dq_rad_s=np.abs(p[:, 1]).max(0).tolist(),
                torque_saturation_fraction=(np.abs(p[:, 3]) >= .98*limits).mean(0).tolist(),
                q_std_rad=p[:, 0].std(0).tolist()))
        save(out / 'substep_summary.json', dict(dof_names=env.dof_names, nominal_cases=probe_stats))
        print('SUBSTEP_SUMMARY', json.dumps(probe_stats), flush=True)
        save(out / 'summary.json', dict(feet=feet, dof_names=env.dof_names, cases=summaries,
            resets=resets, checkpoint_unchanged=True, state_columns=['vx','vy','vz','wx','wy','wz','base_z','tilt_deg']))
        print('RESULT', out, flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
