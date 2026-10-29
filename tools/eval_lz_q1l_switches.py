"""Paired command-transition evaluation using physical, unmodified histories."""
import numpy as np
import torch
from calibrate_lz_q1l_estimator import BLOCK, REPEATS, reset, write_json

STOP = (0., 0., 0.)
LEFT = (0., .4, 0.)
RIGHT = (0., -.4, 0.)
SCENARIOS = [
    ('left_stop_right', [(2., STOP), (4., LEFT), (2., STOP), (4., RIGHT), (2., STOP)]),
    ('left_right', [(2., STOP), (4., LEFT), (4., RIGHT), (4., STOP)]),
    ('right_left', [(2., STOP), (4., RIGHT), (4., LEFT), (4., STOP)]),
    ('left_stop_forward', [(2., STOP), (4., LEFT), (2., STOP), (4., (.5, 0., 0.)), (2., STOP)]),
]


@torch.no_grad()
def evaluate_switches(env, original, candidate, seed, mode, out):
    old_resample = env._resample_commands
    # Resets begin with a genuine standing command; fixed-case reset commands
    # from the steady-state evaluator must not leak into historical frames.
    env._resample_commands = lambda ids: env.commands.__setitem__((ids, slice(None)), 0.)
    try:
        obs = reset(env, seed)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        failed_at = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        steps = round(14. / env.dt)
        schedule = torch.zeros(steps, BLOCK, 3, device=env.device)
        segments = []
        for case_i, (name, parts) in enumerate(SCENARIOS):
            start = 0
            case_segments = []
            for duration, command in parts:
                end = start + round(duration / env.dt)
                schedule[start:end, case_i * REPEATS:(case_i + 1) * REPEATS] = torch.tensor(command, device=env.device)
                case_segments.append((start, end, command))
                start = end
            assert start == steps
            segments.append(case_segments)
        states, masks, before_masks, jumps = [], [], [], []
        for step in range(steps):
            env.commands[:, :3] = schedule[step].repeat(2, 1)
            # Only the newest command changes; older sensor/action history is
            # real physical history, transformed by the policy without mutation.
            obs[:, :3] = env.commands[:, :3] * env.commands_scale
            actions = torch.cat([original.act_inference(obs[:BLOCK]),
                                 candidate.act_inference(obs[BLOCK:])])
            clipped = actions.clamp(-env.cfg.normalization.clip_actions, env.cfg.normalization.clip_actions)
            delta = ((clipped - env.actions) * env.cfg.control.action_scale).abs().max(-1)[0]
            before_masks.append(alive.cpu().numpy().copy())
            obs, _, _, done, _, _, _ = env.step(actions)
            first_failure = alive & done.bool() & ~env.time_out_buf
            failed_at[first_failure] = step
            alive &= ~done.bool()
            tilt = torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.)))
            state = torch.stack([env.base_lin_vel[:, 0], env.base_lin_vel[:, 1],
                                 env.base_ang_vel[:, 2], tilt], -1)
            states.append(state.cpu().numpy().copy())
            masks.append(alive.cpu().numpy().copy())
            jumps.append(delta.cpu().numpy().copy())
        state, valid, active_before, jump = map(np.asarray, [states, masks, before_masks, jumps])
        failures = failed_at.cpu().numpy()
        np.savez_compressed(out / ('switches_seed%d.npz' % seed), state=state,
                            valid=valid, active_before=active_before,
                            max_joint_target_delta_rad=jump, failed_at_step=failures, dt=env.dt)
        rows = []
        for mode_i, label in enumerate(['original', mode]):
            for case_i, (name, _) in enumerate(SCENARIOS):
                first = mode_i * BLOCK + case_i * REPEATS
                sl = slice(first, first + REPEATS)
                row = dict(mode=label, scenario=name, trials=REPEATS,
                           failures=int((failures[sl] >= 0).sum()), segments=[])
                for start, end, command in segments[case_i]:
                    warm = start + round(1. / env.dt)
                    mask = valid[warm:end, sl]
                    samples = state[warm:end, sl][mask]
                    boundary_jumps = jump[start, sl][active_before[start, sl]]
                    part = dict(start_s=start * env.dt, end_s=end * env.dt, command=command,
                                tracking_warmup_s=1., valid_fraction=float(mask.mean()),
                                new_failures=int(((failures[sl] >= start) & (failures[sl] < end)).sum()),
                                boundary_target_delta_rad_max=float(boundary_jumps.max()) if len(boundary_jumps) else None,
                                boundary_target_delta_rad_mean=float(boundary_jumps.mean()) if len(boundary_jumps) else None)
                    if len(samples):
                        part['mean'] = samples.mean(0).tolist()
                        part['mae'] = np.abs(samples[:, :3] - command).mean(0).tolist()
                    row['segments'].append(part)
                rows.append(row)
        result = dict(seed=seed, cases=rows, state_columns=['vx', 'vy', 'yaw_rate', 'tilt_deg'],
                      protocol='16 paired randomized starts per scenario/seed; no histories cleared at command switches; 14 s; censor after first termination; segment tracking omits its first second; count all failures.',
                      action_metric='Max absolute joint target change from previous executed action at each segment boundary; radians; action_scale=0.25 and hip_reduction=1.')
        write_json(out / ('switches_seed%d.json' % seed), result)
        print('SWITCHES', seed, [(r['mode'], r['scenario'], r['failures']) for r in rows], flush=True)
        return result
    finally:
        env._resample_commands = old_resample
