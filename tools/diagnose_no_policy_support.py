"""Current new-lineage physics, constant zero targets, no policy or checkpoint load."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2UnifiedRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict

ROOT = Path(__file__).resolve().parents[1]
perturbations = np.random.RandomState(1).uniform(-.03, .03, (4, 6))
# Mirror polar linear velocities and axial angular velocities across the y=0 plane.
STARTS = np.concatenate([np.zeros((1, 6)), perturbations, perturbations * [1, -1, 1, -1, 1, -1]])

def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

class FixedTargetRobot(LZQ1LV2UnifiedRobot):
    def _resample_commands(self, ids):
        self.commands[ids] = 0.
        self.command_sample_mode[ids] = 0
        self._clear_support(ids)
        self.v2_air_steps[ids] = 0

    def _reset_root_states(self, ids):
        self.root_states[ids] = self.base_init_state
        self.root_states[ids, :3] += self.env_origins[ids]
        self.root_states[ids, 7:13] = torch.as_tensor(STARTS, dtype=torch.float, device=self.device)[ids]
        ids32 = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    def _compute_torques(self, actions):
        torque = super()._compute_torques(actions)
        if hasattr(self, 'probe'):
            self.probe.append(torch.stack((self.dof_pos[0], self.dof_vel[0],
                self.joint_pos_target[0], torque[0])).detach().cpu().numpy().copy())
        return torque

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', required=True, type=Path)
    opts, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    assert args.task == 'lz_q1l_flat_v2_posture'
    out = opts.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    cfg, _ = task_registry.get_cfgs(args.task)
    assert cfg.sim.dt == .001 and cfg.control.decimation == 20
    assert cfg.control.stiffness == {'joint': 80.} and cfg.control.damping == {'joint': 2.}
    assert all(v == 0. for v in cfg.init_state.default_joint_angles.values())
    cfg.env.num_envs = args.num_envs = len(STARTS)
    cfg.env.episode_length_s = 20.
    cfg.commands.resampling_time = 1e6
    cfg.noise.add_noise = False
    for name in dir(cfg.domain_rand):
        if not name.startswith('_') and isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    args.headless = True
    cfg.sim.use_gpu_pipeline = cfg.sim.physx.use_gpu = args.sim_device.startswith('cuda')
    cfg.sim.physx.num_threads = 2
    task_registry.task_classes[args.task] = FixedTargetRobot
    sources = [Path(__file__).resolve()] + [ROOT/p for p in (
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
        'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
        'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
        'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf')]
    (out/'provenance').mkdir()
    for i, path in enumerate(sources): shutil.copyfile(path, out/'provenance'/('%02d_'%i+path.name))
    manifest = dict(task=args.task, checkpoint=None, policy_constructed=False,
        controller='Zero action at every step; unmodified existing PD controller; all 12 targets zero radians',
        starts=STARTS.tolist(), duration_s=12., statistics_window_s=[2.,12.], control_dt_s=.02, physics_dt_s=.001,
        backend=args.sim_device, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        interpretation='Static support diagnostic only, not walking. Fz>1N is contact, collision-sphere bottom gap>5mm plus Fz<=1N is unsupported. Samples after first reset excluded. Substep joint data for nominal start only.')
    save(out/'manifest.json', manifest)
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        assert not env.default_dof_pos.any()
        assert torch.all(env.p_gains == 80) and torch.all(env.d_gains == 2)
        assert torch.all(env.Kp_factors == 1) and torch.all(env.Kd_factors == 1)
        body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        manifest.update(feet=[body_names[i] for i in env.feet_indices.tolist()], dof_names=env.dof_names,
            torque_limits_Nm=env.torque_limits.cpu().tolist())
        save(out/'manifest.json', manifest); save(out/'config.json', class_to_dict(cfg))
        env.reset(); env.probe = []
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        history = {k: [] for k in ('valid','fz','gap','q','dq','target','torque','height','tilt')}
        resets = []
        with torch.no_grad():
            for step in range(600):
                _, _, _, done, _, _, _ = env.step(actions)
                assert not env.joint_pos_target.any()
                for index in (alive & done.bool()).nonzero().flatten().tolist():
                    resets.append(dict(index=index, time_s=(step+1)*env.dt))
                alive &= ~done.bool()
                values = dict(valid=alive, fz=env.contact_forces[:,env.feet_indices,2], gap=env.support_foot_gap,
                    q=env.dof_pos, dq=env.dof_vel, target=env.joint_pos_target, torque=env.torques,
                    height=env.root_states[:,2], tilt=torch.rad2deg(torch.acos((-env.projected_gravity[:,2]).clamp(-1,1))))
                for k, v in values.items(): history[k].append(v.cpu().numpy().copy())
                if (step+1)%100 == 0: print('STEPS',step+1,flush=True)
        data = {k: np.asarray(v) for k,v in history.items()}
        assert all(np.isfinite(v).all() for v in data.values())
        probe = np.asarray(env.probe)
        assert len(probe) == 12000 and np.isfinite(probe).all() and not probe[:,2].any()
        np.savez_compressed(out/'traces.npz', **data)
        np.savez_compressed(out/'substeps.npz', data=probe, dt=.001, columns=['q','dq','target','torque'])
        groups = [('nominal',[0]),('perturbed',list(range(1,9)))] + [('trial_%d'%i,[i]) for i in range(1,9)]
        summaries = []
        for name, ids in groups:
            valid = data['valid'][100:,ids]
            row = dict(group=name, trials=len(ids), failures=sum(r['index'] in ids for r in resets), valid_fraction=float(valid.mean()))
            if valid.any():
                def sample(key): return data[key][100:,ids][valid]
                fz, gap = sample('fz'), sample('gap')
                row.update(contact_fraction=(fz>1).mean(0).tolist(), mean_force_N=fz.mean(0).tolist(),
                    mean_gap_mm=(gap.mean(0)*1000).tolist(), max_gap_mm=(gap.max(0)*1000).tolist(),
                    unsupported_fraction=((fz<=1)&(gap>.005)).mean(0).tolist(),
                    all_four_contact_fraction=float((fz>1).all(1).mean()),
                    joint_error_mean_deg=(np.abs(sample('q')-sample('target')).mean(0)*180/np.pi).tolist(),
                    height_mean_m=float(sample('height').mean()), tilt_mean_deg=float(sample('tilt').mean()))
            summaries.append(row)
        p = probe[2000:]
        sub = dict(q_std_deg=(p[:,0].std(0)*180/np.pi).tolist(), mean_abs_velocity_rad_s=np.abs(p[:,1]).mean(0).tolist(),
            peak_abs_velocity_rad_s=np.abs(p[:,1]).max(0).tolist(),
            torque_saturation_fraction=(np.abs(p[:,3])>=.98*env.torque_limits.cpu().numpy()).mean(0).tolist())
        report = dict(feet=manifest['feet'], dof_names=env.dof_names, cases=summaries, resets=resets, nominal_substeps=sub,
            all_targets_zero=True, checkpoint_loaded=False, network_called=False)
        save(out/'summary.json', report)
        print('SUMMARY',json.dumps(report),flush=True)
    finally:
        if env is not None: env.gym.destroy_sim(env.sim)

if __name__ == '__main__': main()
