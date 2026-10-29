"""CPU integration checks for new V2; no learning updates or checkpoint writes."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import isaacgym  # Must precede torch.
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from check_lz_q1l_v1_support import check_support_rules
from train_lz_q1l_v1_support_500 import differences
from train_lz_q1l_v2_unified import TASK, ROOT, SOURCE, SOURCE_SHA256, load_new_v1_weights, install_logging


def check_saver():
    class FakeRunner:
        current_learning_iteration = 0
        writer = SimpleNamespace(flush=lambda: None, add_scalar=lambda *a: None)
        env = SimpleNamespace(take_diagnostics=lambda: {})

        def log(self, locs):
            pass

        def save(self, path, infos=None):
            Path(path).write_text(str(self.current_learning_iteration))

    with tempfile.TemporaryDirectory(prefix='v2_checkpoint_hook_') as directory:
        out = Path(directory)
        runner = FakeRunner()
        install_logging(runner, out, {}, 51)
        for i in range(51):
            runner.log(dict(it=i, mean_value_loss=0., mean_surrogate_loss=0., mean_estimation_loss=0., mean_swap_loss=0.))
            runner.save('legacy_save_must_be_disabled')
        assert sorted(p.name for p in out.glob('model_*.pt')) == ['model_50.pt', 'model_51.pt']
        assert (out / 'model_50.pt').read_text() == '50'
        assert (out / 'model_51.pt').read_text() == '51'
        assert runner.current_learning_iteration == 0
    return 'passed: completed-update 50, final 51, legacy saver suppressed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    opts = parser.parse_args()
    opts.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cfg, train = task_registry.get_cfgs(TASK)
    v1, v1_train = task_registry.get_cfgs('lz_q1l_flat_v1_support')
    env_diff = differences(class_to_dict(v1), class_to_dict(cfg))
    train_diff = differences(class_to_dict(v1_train), class_to_dict(train))
    assert set(env_diff) == {
        'commands.curriculum', 'commands.pure_lateral_prob', 'commands.pure_yaw_prob',
        'commands.pure_lateral_speed_range', 'commands.pure_yaw_speed_range',
        'commands.ranges.lin_vel_x', 'commands.ranges.lin_vel_y', 'commands.ranges.ang_vel_yaw',
        'rewards.lateral_tracking_sigma', 'rewards.scales.tracking_lateral'}
    assert set(train_diff) == {'runner.experiment_name', 'runner.max_iterations'}
    assert train.runner.max_iterations == 800 and train.runner.save_interval == 50
    previous = json.loads(Path('/tmp/lz_q1l_merge_1ms_before.json').read_text()) if Path('/tmp/lz_q1l_merge_1ms_before.json').exists() else None
    if previous:
        for name, expected in previous.items():
            assert [class_to_dict(c) for c in task_registry.get_cfgs(name)] == expected, name
    assert cfg.sim.dt == .001 and cfg.control.decimation == 20
    assert cfg.commands.zero_command_prob == .2 and cfg.commands.pure_forward_prob == .3
    assert cfg.rewards.scales.stand_unsupported == -1.
    source_tree = ast.parse((ROOT / 'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py').read_text())
    archive_tree = ast.parse((ROOT / 'media/v1_support_stand_forward_20261024/run_20261024_154403_9ogp/result/provenance/5_lz_q1l_v1_support.py').read_text())
    get_v1 = lambda tree: next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LZQ1LV1SupportRobot')
    assert ast.dump(get_v1(source_tree)) == ast.dump(get_v1(archive_tree)), 'New V1 logic changed'
    default_config = class_to_dict(cfg)
    cfg = copy.deepcopy(cfg)
    cfg.env.num_envs = 24
    cfg.noise.add_noise = False
    cfg.commands.resampling_time = 1e6
    cfg.sim.use_gpu_pipeline = cfg.sim.physx.use_gpu = False
    cfg.sim.physx.num_threads = 2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    sys.argv = [sys.argv[0], '--task', TASK, '--headless', '--seed', '1',
                '--sim_device', 'cpu', '--rl_device', 'cpu', '--pipeline', 'cpu', '--num_threads', '2']
    args = get_args()
    env = None
    try:
        env, _ = task_registry.make_env(TASK, args=args, env_cfg=cfg)
        # Initialize BaseTask before scripted quaternion helpers (legacy JIT).
        support_checks = check_support_rules()
        ids = torch.arange(env.num_envs)
        counts = torch.zeros(6, dtype=torch.long)
        for _ in range(2000):
            env._resample_commands(ids)
            modes, commands = env.command_sample_mode, env.commands
            counts += torch.bincount(modes, minlength=6)
            assert (commands[modes == 0] == 0).all()
            for mode, axis, lo, hi, sign in ((1, 0, .2, .5, 1), (2, 1, .15, .25, 1),
                    (3, 1, .15, .25, -1), (4, 2, .3, .5, 1), (5, 2, .3, .5, -1)):
                rows = commands[modes == mode]
                signed = rows[:, axis] * sign
                assert ((signed >= lo) & (signed <= hi)).all()
                assert (rows[:, [i for i in range(4) if i != axis]] == 0).all()
        proportions = counts.float() / counts.sum()
        assert (proportions - torch.tensor([.2, .3, .125, .125, .125, .125])).abs().max() < .012
        # Subset resampling must not alter other environments, including timers.
        env.support_offground_steps[:] = 7
        saved = env.commands.clone()
        env._resample_commands(ids[:2])
        assert torch.equal(env.commands[2:], saved[2:])
        assert (env.support_offground_steps[2:] == 7).all() and not env.support_offground_steps[:2].any()
        # Extra reward gates out stand/forward/yaw/mixed; left/right symmetric.
        env.commands.zero_()
        env.base_lin_vel.zero_()
        env.commands[:6, :3] = torch.tensor([[0,.2,0],[0,-.2,0],[.3,0,0],[0,0,.4],[.3,.2,0],[0,.2,.4]])
        reward = env._reward_tracking_lateral()
        assert torch.allclose(reward[:2], torch.full((2,), torch.exp(torch.tensor(-1.)).item()))
        assert not reward[2:].any()
        env.base_lin_vel[:2, 1] = env.commands[:2, 1]
        assert torch.equal(env._reward_tracking_lateral()[:2], torch.ones(2))
        runner = HIMOnPolicyRunner(env, class_to_dict(train), log_dir=None, device='cpu')
        state = load_new_v1_weights(runner)
        assert all(torch.equal(v, state['model_state_dict'][k]) for k,v in runner.alg.actor_critic.state_dict().items())
        before = {k: v.clone() for k,v in runner.alg.actor_critic.state_dict().items()}
        runner.alg.actor_critic.eval()
        obs, _ = env.reset()
        env.take_diagnostics()
        fixed_modes = ids % 6
        fixed_commands = torch.tensor([[0.,0.,0.],[.35,0.,0.],[0.,.2,0.],[0.,-.2,0.],[0.,0.,.4],[0.,0.,-.4]])
        with torch.no_grad():
            for step in range(150):
                env.commands.zero_()
                env.commands[:, :3] = fixed_commands[fixed_modes]
                env.command_sample_mode[:] = fixed_modes
                obs[:, :3] = env.commands[:, :3] * env.commands_scale
                # Explicit raw new-V1 baseline: integration, not demonstration.
                actions = runner.alg.actor_critic.act_inference(obs)
                obs, privileged, reward, done, _, _, _ = env.step(actions)
                assert all(torch.isfinite(x).all() for x in (actions, obs, privileged, reward, env.support_foot_gap))
                if step % 50 == 49:
                    print('SIM_STEP', step + 1, flush=True)
        diagnostics = env.take_diagnostics()
        assert set(diagnostics) == set(env.command_names)
        assert sum(r['samples'] for r in diagnostics.values()) == 150 * env.num_envs
        assert all(r['samples'] == 600 for r in diagnostics.values())
        assert not env.take_diagnostics(), 'Diagnostics drain must clear accumulated sums'
        assert all(torch.equal(v, before[k]) for k,v in runner.alg.actor_critic.state_dict().items())
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
        env.support_offground_steps[:] = 4
        env.v2_air_steps[:] = 5
        env.reset_idx(ids[:2])
        assert not env.support_offground_steps[:2].any() and not env.v2_air_steps[:2].any()
        assert (env.v2_air_steps[2:] == 5).all()
        saver = check_saver()
        feet = [env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])[i] for i in env.feet_indices.tolist()]
        report = dict(status='passed_no_training', env_diff=env_diff, train_diff=train_diff,
            default_config=default_config, command_draws=int(counts.sum()), proportions=proportions.tolist(),
            modes=env.command_names, feet=feet, support_checks=support_checks, checkpoint_hook=saver,
            checkpoint=str(SOURCE), checkpoint_sha256=SOURCE_SHA256,
            checkpoint_loading='strict full state_dict; fresh PPO/estimator optimizers; no weight changes',
            simulation=dict(backend='CPU PhysX', environments=24, steps=150, dt=.001, decimation=20),
            smoke_diagnostics=diagnostics,
            limitations='3 seconds per environment using raw new V1 weights, without learning. Checks implementation, not V2 skill, convergence, or GPU training stability.')
        (opts.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print('PASS', opts.out, 'proportions', proportions.tolist(), flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
