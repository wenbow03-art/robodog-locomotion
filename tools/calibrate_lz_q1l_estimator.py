"""Bounded V8 estimator calibration; actor, std and latent features stay frozen.

The three velocity-output rows alone are optimized with next-state velocity
supervision. Deterministic and sampled replay are balanced by command/mode.
Every checkpoint is evaluated against the original in paired nominal resets.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import isaacgym  # noqa: F401
from isaacgym import gymtorch
import numpy as np
import torch
from torch import nn

from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l import LZQ1LRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules import HIMActorCritic

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'HIMLoco/legged_gym/logs1/flat_lz_q1l_v8/v8_ft_v7_200_200iters_20261018_130426_2712609/model_200.pt'
CASES = [('stand', (0., 0., 0.)), ('forward_04', (.4, 0., 0.)),
         ('forward_05', (.5, 0., 0.)), ('left_02', (0., .2, 0.)),
         ('right_02', (0., -.2, 0.)), ('left_04', (0., .4, 0.)),
         ('right_04', (0., -.4, 0.)), ('diagonal_left', (.3, .4, 0.)),
         ('diagonal_right', (.3, -.4, 0.)), ('yaw_left', (0., 0., .8)),
         ('yaw_right', (0., 0., -.8))]
REPEATS = 16
BLOCK = REPEATS * len(CASES)
DURATION = 12.
WARMUP = 2.


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


class CalibrationRobot(LZQ1LRobot):
    """Pair initial root velocities across the two policy blocks."""
    reset_velocities = np.zeros((BLOCK, 6), dtype=np.float32)

    def _resample_commands(self, env_ids):
        commands = torch.tensor([v for _, v in CASES], device=self.device)
        self.commands[env_ids, :3] = commands[(env_ids % BLOCK) // REPEATS]
        self.commands[env_ids, 3] = 0.

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        self.root_states[env_ids, 7:13] = torch.as_tensor(
            self.reset_velocities, device=self.device)[env_ids % BLOCK]
        ids = env_ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids), len(ids))


def reset(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    CalibrationRobot.reset_velocities = np.random.RandomState(seed).uniform(-.5, .5, (BLOCK, 6)).astype(np.float32)
    env.obs_buf.zero_()
    env.privileged_obs_buf.zero_()
    env.last_contacts.zero_()
    env.actions.zero_()
    env.common_step_counter = 0
    obs, _ = env.reset()
    assert torch.equal(env.commands[:BLOCK], env.commands[BLOCK:])
    return obs


def make_actor(env, train_cfg, state):
    actor = HIMActorCritic(env.num_obs, env.num_privileged_obs,
                          env.num_one_step_obs, env.num_actions,
                          **class_to_dict(train_cfg.policy)).to(env.device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


@torch.no_grad()
def collect(env, actor, seed, out):
    obs = reset(env, seed)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    history, targets, groups = [], [], []
    group = torch.arange(env.num_envs, device=env.device) // REPEATS
    for step in range(round(DURATION / env.dt)):
        before = obs.clone()
        actions = actor.act_inference(before)
        actions[BLOCK:] += torch.randn_like(actions[BLOCK:]) * actor.std
        obs, critic_obs, _, done, _, _, _ = env.step(actions)
        alive &= ~done.bool()
        # Match HIMEstimator.update: history at t predicts true velocity at t+1.
        target = critic_obs[:, env.num_one_step_obs:env.num_one_step_obs + 3]
        assert torch.allclose(target[alive], env.base_lin_vel[alive] * env.obs_scales.lin_vel)
        if step >= env.history_length and step % 2 == 0 and alive.any():
            history.append(before[alive].cpu())
            targets.append(target[alive].cpu())
            groups.append(group[alive].cpu())
        if (step + 1) % 200 == 0:
            print('COLLECT', seed, step + 1, flush=True)
    dataset = {'history': torch.cat(history), 'target_scaled_velocity': torch.cat(targets),
               'group': torch.cat(groups), 'seed': seed,
               'group_definition': '0..10 deterministic, 11..21 sampled, CASES order',
               'target': 'next-state body velocity * obs_scales.lin_vel; termination transitions excluded'}
    torch.save(dataset, out)
    print('DATASET', seed, len(dataset['history']), flush=True)
    return dataset


@torch.no_grad()
def evaluate(env, baseline, candidate, seed, label, out):
    obs = reset(env, seed)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(alive)
    records, masks, predicted = [], [], []
    for step in range(round(DURATION / env.dt)):
        actions = torch.cat((baseline.act_inference(obs[:BLOCK]),
                             candidate.act_inference(obs[BLOCK:])), 0)
        estimate = torch.cat((baseline.estimator(obs[:BLOCK])[0],
                              candidate.estimator(obs[BLOCK:])[0]), 0) / env.obs_scales.lin_vel
        obs, _, _, done, _, _, _ = env.step(actions)
        failed |= alive & done.bool() & ~env.time_out_buf
        alive &= ~done.bool()
        tilt = torch.rad2deg(torch.acos((-env.projected_gravity[:, 2]).clamp(-1., 1.)))
        state = torch.stack((env.base_lin_vel[:, 0], env.base_lin_vel[:, 1],
                             env.base_ang_vel[:, 2], env.root_states[:, 2], tilt), -1)
        records.append(state.cpu().numpy().copy())
        masks.append(alive.cpu().numpy().copy())
        predicted.append(estimate.cpu().numpy().copy())
    state, valid, estimate = np.asarray(records), np.asarray(masks), np.asarray(predicted)
    np.savez_compressed(out / ('eval_%s_seed%d.npz' % (label, seed)),
                        state=state, valid=valid, estimated_velocity=estimate, dt=env.dt)
    start = round(WARMUP / env.dt)
    result = {'seed': seed, 'label': label, 'repeats': REPEATS, 'cases': []}
    failure_counts = failed.cpu().numpy()
    for mode_i, mode in enumerate(['baseline', 'calibrated']):
        for case_i, (name, command) in enumerate(CASES):
            first = mode_i * BLOCK + case_i * REPEATS
            sl = slice(first, first + REPEATS)
            mask = valid[start:, sl]
            samples = state[start:, sl][mask]
            row = dict(mode=mode, case=name, command=command,
                       failures=int(failure_counts[sl].sum()), valid_fraction=float(mask.mean()))
            if len(samples):
                row.update(mean=samples.mean(0).tolist(),
                           mae=np.abs(samples[:, :3] - command).mean(0).tolist(),
                           estimated_vy_mean=float(estimate[start:, sl][mask][:, 1].mean()),
                           predicted_next_vy_mae=float(np.abs(estimate[start:, sl][mask][:, 1] - samples[:, 1]).mean()),
                           tilt_p95_deg=float(np.percentile(samples[:, 4], 95)))
            result['cases'].append(row)
    write_json(out / ('eval_%s_seed%d.json' % (label, seed)), result)
    print('EVAL', label, seed, json.dumps([
        {k: c.get(k) for k in ['mode', 'case', 'mean', 'failures', 'estimated_vy_mean']}
        for c in result['cases'] if c['case'] in ['left_02', 'left_04', 'right_04', 'forward_05']]), flush=True)
    return result


def gate(results):
    regressions, improvements, all_target_pass = [], [], True
    for result in results:
        base = {r['case']: r for r in result['cases'] if r['mode'] == 'baseline'}
        for c in [r for r in result['cases'] if r['mode'] == 'calibrated']:
            b = base[c['case']]
            if 'mae' not in b or 'mae' not in c:
                regressions.append([result['seed'], c['case'], 'no_valid_samples'])
                all_target_pass = False
                continue
            if c['failures'] > b['failures']:
                regressions.append([result['seed'], c['case'], 'increased_failures'])
            for axis, tolerance in enumerate([.03, .03, .05]):
                if c['mae'][axis] > b['mae'][axis] + tolerance:
                    regressions.append([result['seed'], c['case'], 'mae_axis_%d' % axis])
            if c['case'] in ['left_02', 'left_04']:
                improvements.append(b['mae'][1] - c['mae'][1])
                if c['failures']:
                    all_target_pass = False
            if c['case'] in ['left_02', 'left_04', 'right_02', 'right_04']:
                all_target_pass &= c['failures'] == 0 and c['mae'][1] <= .1 and c['mae'][2] <= .2
    improved = bool(improvements) and min(improvements) >= .03
    return dict(relative_gate_pass=not regressions and improved,
                lateral_target_pass=bool(all_target_pass), regressions=regressions,
                left_mae_improvement_mean=float(np.mean(improvements)) if improvements else None)


@torch.no_grad()
def features(actor, dataset, device):
    output = []
    for batch in dataset['history'].split(4096):
        output.append(actor.estimator.encoder[:-1](batch.to(device)))
    return torch.cat(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    opts = parser.parse_args()
    out = opts.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        raise FileExistsError('Refusing to reuse an experiment directory')
    started = time.time()
    stages = [25, 75, 150, 300]
    manifest = dict(source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    method='Supervised velocity-head-only calibration; frozen encoder features, latent, actor, critic and action std',
                    learning_rate=opts.learning_rate, batch_size=2048, stages=stages,
                    data_seeds={'train': 101, 'validation': 202}, evaluation_seeds=[1, 2], confirmation_seeds=[3, 4],
                    duration=DURATION, warmup=WARMUP, repeats=REPEATS, cases=CASES,
                    sampling='Equal probability for each of 22 command/mode groups; fixed-command replay',
                    success_gate='Per-case no extra failures or MAE regression >[.03,.03,.05]; both positive lateral speeds improve MAE by >=.03 on each evaluation seed.',
                    limitations=['Nominal flat ground only; no deployment robustness claim.',
                                 'Mean velocity/MAE censor samples after first failure; failures reported separately.',
                                 'Paired root reset velocities within evaluation; PhysX contact caches may introduce small differences.',
                                 'Velocity-head-only calibration does not test all possible estimator architectures.'],
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid())
    sources = [Path(__file__), ROOT / 'env.sh',
               ROOT / 'tools/run_lz_q1l_estimator_calibration.sh',
               ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
               ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
               ROOT / 'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
               ROOT / 'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf',
               ROOT / 'HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py',
               ROOT / 'HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py']
    (out / 'provenance').mkdir()
    manifest['source_sha256s'] = {}
    for path in sources:
        manifest['source_sha256s'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, out / 'provenance' / path.name)
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'status.json', {'state': 'initializing'})
    torch.manual_seed(0)
    witness = torch.randn(8, 8, device='cuda:0')
    assert torch.isfinite(witness @ witness).all()
    print('CUDA_WITNESS', torch.cuda.get_device_name(0), torch.__version__, flush=True)
    sys.argv = [sys.argv[0], '--task', 'lz_q1l_flat_v8', '--headless', '--seed', '101',
                '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
    args = get_args()
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = args.num_envs = 2 * BLOCK
    cfg.env.episode_length_s = 17.
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.noise.add_noise = False
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    task_registry.task_classes[args.task] = CalibrationRobot
    write_json(out / 'env_config.json', class_to_dict(cfg))
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        source = torch.load(SOURCE, map_location='cpu')
        original = make_actor(env, train_cfg, source['model_state_dict'])
        candidate = make_actor(env, train_cfg, source['model_state_dict'])
        baseline_state = {k: v.detach().cpu().clone() for k, v in original.state_dict().items()}
        write_json(out / 'status.json', {'state': 'collecting_training_data'})
        train = collect(env, original, 101, out / 'train_replay.pt')
        write_json(out / 'status.json', {'state': 'collecting_validation_data'})
        validation = collect(env, original, 202, out / 'validation_replay.pt')
        x = features(original, train, env.device)
        vx = features(original, validation, env.device)
        y = train['target_scaled_velocity'].to(env.device)
        vy = validation['target_scaled_velocity'].to(env.device)
        group_ids = train['group'].to(env.device)
        buckets = [(group_ids == i).nonzero(as_tuple=True)[0] for i in range(2 * len(CASES))]
        assert all(len(b) > 0 for b in buckets)
        final_layer = candidate.estimator.encoder[-1]
        head = nn.Linear(final_layer.in_features, 3).to(env.device)
        with torch.no_grad():
            head.weight.copy_(final_layer.weight[:3])
            head.bias.copy_(final_layer.bias[:3])
        optimizer = torch.optim.Adam(head.parameters(), lr=opts.learning_rate)
        torch.manual_seed(303)
        logs, assessments = [], []
        initial_mae = ((head(vx) - vy).abs().mean(0) / env.obs_scales.lin_vel).detach().cpu().tolist()
        write_json(out / 'initial_validation.json', {'velocity_mae_mps': initial_mae})
        for update in range(1, stages[-1] + 1):
            indices = torch.cat([b[torch.randint(len(b), (94,), device=env.device)] for b in buckets])
            indices = indices[torch.randperm(len(indices), device=env.device)[:2048]]
            prediction = head(x[indices])
            loss = (prediction - y[indices]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite calibration loss')
            if update in stages:
                with torch.no_grad():
                    final_layer.weight[:3].copy_(head.weight)
                    final_layer.bias[:3].copy_(head.bias)
                    current_state = candidate.state_dict()
                    allowed = ['estimator.encoder.%d.weight' % (len(candidate.estimator.encoder)-1),
                               'estimator.encoder.%d.bias' % (len(candidate.estimator.encoder)-1)]
                    for key, tensor in current_state.items():
                        now = tensor.detach().cpu()
                        if key in allowed:
                            assert torch.equal(now[3:], baseline_state[key][3:])
                        else:
                            assert torch.equal(now, baseline_state[key]), key
                    probe = train['history'][:1024].to(env.device)
                    assert torch.equal(original.estimator(probe)[1], candidate.estimator(probe)[1])
                    metrics = dict(update=update, train_loss_scaled=float(loss),
                                   validation_velocity_mae_mps=((head(vx)-vy).abs().mean(0)/env.obs_scales.lin_vel).cpu().tolist(),
                                   latent_bitwise_unchanged=True, actor_and_std_bitwise_unchanged=True)
                logs.append(metrics)
                write_json(out / 'training_metrics.json', logs)
                checkpoint = copy.deepcopy(source)
                checkpoint['model_state_dict'] = {k: v.detach().cpu().clone() for k, v in current_state.items()}
                checkpoint['estimator_optimizer_state_dict'] = candidate.estimator.optimizer.state_dict()
                checkpoint['calibration_optimizer_state_dict'] = optimizer.state_dict()
                checkpoint['infos'] = dict(calibration_updates=update, parent=str(SOURCE),
                                           calibration_method=manifest['method'], resume_with_optimizer_reset=True)
                path = out / ('calibrated_%03d.pt' % update)
                torch.save(checkpoint, path)
                write_json(out / 'status.json', {'state': 'evaluating', 'updates': update})
                rng_state = torch.cuda.get_rng_state()
                evaluations = [evaluate(env, original, candidate, seed, '%03d' % update, out) for seed in [1, 2]]
                torch.cuda.set_rng_state(rng_state)
                decision = dict(update=update, checkpoint=str(path), **gate(evaluations))
                assessments.append(decision)
                write_json(out / 'assessments.json', assessments)
                print('ASSESSMENT', json.dumps(decision), flush=True)
        passing = [d for d in assessments if d['relative_gate_pass']]
        selected = max(passing, key=lambda d: d['left_mae_improvement_mean']) if passing else None
        confirmation = None
        if selected:
            candidate.load_state_dict(torch.load(selected['checkpoint'], map_location=env.device)['model_state_dict'])
            evaluations = [evaluate(env, original, candidate, seed, 'confirmation', out) for seed in [3, 4]]
            confirmation = gate(evaluations)
        outcome = dict(state='completed', elapsed_seconds=time.time()-started,
                       selected_candidate=selected, confirmation=confirmation,
                       promote=bool(selected and confirmation['relative_gate_pass']),
                       source_checkpoint_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest() == manifest['source_sha256'],
                       assessments=assessments)
        assert outcome['source_checkpoint_unchanged']
        write_json(out / 'result.json', outcome)
        write_json(out / 'status.json', outcome)
        print('COMPLETE', json.dumps(outcome), flush=True)
    except BaseException as error:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(error)})
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
