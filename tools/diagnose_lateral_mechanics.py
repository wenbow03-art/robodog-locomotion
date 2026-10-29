"""Evaluate fixed commands and gait mechanics without changing training rewards.

Compare deterministic deployment, sampled actions, and an oracle velocity input.
Each mode/case has 16 initializations, censored after its first termination.
"""
import hashlib
import json
import sys
from pathlib import Path

import isaacgym  # noqa: F401 -- must precede torch
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l import LZQ1LRobot
from legged_gym.utils.log_paths import get_experiment_log_dir
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict


CASES = [("stand", (0., 0., 0.)), ("forward_04", (.4, 0., 0.)),
         ("forward_05", (.5, 0., 0.)), ("left_02", (0., .2, 0.)),
         ("right_02", (0., -.2, 0.)), ("left_04", (0., .4, 0.)),
         ("right_04", (0., -.4, 0.)), ("diagonal_left", (.3, .4, 0.)),
         ("diagonal_right", (.3, -.4, 0.))]
MODES = ["deterministic", "sampled", "oracle_velocity"]
REPEATS = 16
DURATION, WARMUP = 12., 2.


def fixed_commands(self, env_ids):
    commands = torch.tensor([cmd for _, cmd in CASES], device=self.device)
    self.commands[env_ids, :3] = commands[(env_ids // REPEATS) % len(CASES)]
    self.commands[env_ids, 3] = 0.


def main():
    idx = sys.argv.index("--out")
    out = Path(sys.argv[idx + 1])
    del sys.argv[idx:idx + 2]
    if out.exists() or Path(str(out) + ".npz").exists():
        raise FileExistsError(out)
    args = get_args()
    args.headless = True
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    args.num_envs = cfg.env.num_envs = len(MODES) * len(CASES) * REPEATS
    cfg.env.episode_length_s = DURATION + 5.
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.noise.add_noise = False
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    LZQ1LRobot._resample_commands = fixed_commands
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    try:
        train_cfg.runner.resume = True
        runner, _ = task_registry.make_alg_runner(env, args.task, args=args, train_cfg=train_cfg)
        actor = runner.alg.actor_critic
        # This historical ablation intentionally compares raw checkpoint modes,
        # not the default left-mirrored demonstration controller.
        actor.eval()
        policy = actor.act_inference
        checkpoint = Path(get_experiment_log_dir(train_cfg)) / args.load_run / ('model_%d.pt' % args.checkpoint)
        reward_names = list(env.reward_names)
        latest_rewards = {}
        # Capture the exact calls made by compute_reward: never invoke stateful
        # rewards (notably feet_air_time) a second time.
        for i, (name, function) in enumerate(zip(reward_names, env.reward_functions)):
            def capture(name=name, function=function):
                raw = function()
                latest_rewards[name] = raw.detach().clone() * env.reward_scales[name]
                return raw
            env.reward_functions[i] = capture
        env.obs_buf.zero_()
        obs, _ = env.reset()
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        failures = torch.zeros_like(alive)
        records = {key: [] for key in ['state', 'valid', 'contact', 'dof_pos',
                   'dof_vel', 'torque_ratio', 'reward_rate', 'estimated_velocity']}
        previous_contact = env.contact_forces[:, env.feet_indices, 2] > 1.
        sampled_start = len(CASES) * REPEATS
        oracle_start = 2 * sampled_start
        with torch.inference_mode():
            for step in range(round(DURATION / env.dt)):
                estimate, latent = actor.estimator(obs)
                actions = policy(obs)
                actions[sampled_start:oracle_start] += torch.randn_like(actions[sampled_start:oracle_start]) * actor.std
                # Simulation-only diagnostic: keep the estimator's latent vector
                # but replace its velocity channels with current ground truth.
                oracle_input = torch.cat((obs[oracle_start:, :env.num_one_step_obs],
                                          env.base_lin_vel[oracle_start:] * env.obs_scales.lin_vel,
                                          latent[oracle_start:]), dim=-1)
                actions[oracle_start:] = actor.actor(oracle_input)
                estimate = estimate / env.obs_scales.lin_vel
                obs, _, _, dones, _, _, _ = env.step(actions)
                failures |= alive & dones.bool() & ~env.time_out_buf
                alive &= ~dones.bool()
                rewards = torch.stack([latest_rewards[n] for n in reward_names], -1)
                if alive.any() and not torch.allclose(rewards.sum(-1)[alive], env.rew_buf[alive], atol=1e-5):
                    raise RuntimeError('Captured rewards do not match environment rewards')
                contact = env.contact_forces[:, env.feet_indices, 2] > 1.
                filtered_contact = contact | previous_contact
                previous_contact = contact.clone()
                tilt = torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.)))
                values = dict(
                    state=torch.stack([env.base_lin_vel[:, 0], env.base_lin_vel[:, 1],
                                       env.base_ang_vel[:, 2], env.root_states[:, 2], tilt], -1),
                    valid=alive, contact=filtered_contact, dof_pos=env.dof_pos,
                    dof_vel=env.dof_vel, torque_ratio=env.torques.abs() / env.torque_limits,
                    reward_rate=rewards / env.dt, estimated_velocity=estimate)
                for key, value in values.items():
                    records[key].append(value.cpu().numpy().copy())
                if (step + 1) % 100 == 0:
                    print('Evaluation steps:', step + 1, flush=True)
        data = {key: np.asarray(value) for key, value in records.items()}
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out) + '.npz', **data, dt=env.dt)
        hips = [i for i, name in enumerate(env.dof_names) if name.startswith('roll_')]
        result = dict(task=args.task, checkpoint=str(checkpoint),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      seed=args.seed, repeats=REPEATS, duration=DURATION, warmup=WARMUP,
                      protocol='Flat ground, all domain randomization and observation noise disabled. Random initial root velocities retained. Independent initializations per mode. Post-first-termination and initial 2 s excluded. Torque ratios sampled at final physics substep only.',
                      limitations='One RNG seed; randomized initial states are not paired across policy modes. Reward contributions are observational, not causal ablations. Estimator output predicts next-state velocity; oracle uses current velocity and retains the learned latent. Oracle input is simulation-only and not a deployment fix.',
                      env_config=class_to_dict(cfg), dof_names=list(env.dof_names),
                      feet_names=[env.body_names[i] for i in env.feet_indices.cpu().tolist()] if hasattr(env, 'body_names') else env.feet_indices.cpu().tolist(),
                      action_noise_std=actor.std.detach().cpu().tolist(),
                      reward_names=reward_names, state_columns=['vx', 'vy', 'wz', 'height', 'tilt_deg'], cases=[])
        start = round(WARMUP / env.dt)
        failed = failures.cpu().numpy()
        for mode_i, mode in enumerate(MODES):
            for case_i, (name, command) in enumerate(CASES):
                first = (mode_i * len(CASES) + case_i) * REPEATS
                sl = slice(first, first + REPEATS)
                valid = data['valid'][start:, sl]
                samples = data['state'][start:, sl][valid]
                row = dict(mode=mode, case=name, command=command, failures=int(failed[sl].sum()), valid_fraction=float(valid.mean()))
                if len(samples):
                    contacts = data['contact'][start:, sl]
                    pos = data['dof_pos'][start:, sl]
                    temporal_std = []
                    for e in range(REPEATS):
                        if valid[:, e].sum() >= 2:
                            temporal_std.append(pos[valid[:, e], e][:, hips].std(0).mean())
                    # Only count complete swings starting after warmup. Contact
                    # filtering matches the reward's two-step contact filter.
                    swings = []
                    landings = 0
                    for e in range(REPEATS):
                        for foot in range(contacts.shape[-1]):
                            swing_start = None
                            for t in range(1, len(contacts)):
                                if not valid[t, e]:
                                    break
                                if contacts[t-1, e, foot] and not contacts[t, e, foot]:
                                    swing_start = t
                                if not contacts[t-1, e, foot] and contacts[t, e, foot] and swing_start is not None:
                                    swings.append((t - swing_start) * env.dt)
                                    landings += 1
                                    swing_start = None
                    row.update(mean=samples.mean(0).tolist(),
                               mae=np.abs(samples[:, :3] - command).mean(0).tolist(),
                               tilt_p95_deg=float(np.percentile(samples[:, 4], 95)),
                               all_feet_contact_fraction=float(contacts.all(-1)[valid].mean()),
                               foot_contact_fraction=contacts[valid].mean(0).tolist(),
                               hip_temporal_std_rad=float(np.mean(temporal_std)) if temporal_std else None,
                               hip_speed_abs_mean=float(np.abs(data['dof_vel'][start:, sl][valid][:, hips]).mean()),
                               hip_torque_saturation_fraction=float((data['torque_ratio'][start:, sl][valid][:, hips] > .98).mean()),
                               complete_swing_count=landings,
                               landings_per_robot_second=landings / (valid.sum() * env.dt),
                               swing_duration_mean=float(np.mean(swings)) if swings else None,
                               swing_under_03_fraction=float(np.mean(np.array(swings) < .3)) if swings else None,
                               estimated_vy_mean=float(data['estimated_velocity'][start:, sl][valid][:, 1].mean()),
                               predicted_next_vy_mae=float(np.abs(data['estimated_velocity'][start:, sl][valid][:, 1] - samples[:, 1]).mean()),
                               reward_rate=dict(zip(reward_names, data['reward_rate'][start:, sl][valid].mean(0).tolist())))
                result['cases'].append(row)
                print(json.dumps(row), flush=True)
        out.write_text(json.dumps(result, indent=2) + '\n')
        print('SAVED', out, flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
