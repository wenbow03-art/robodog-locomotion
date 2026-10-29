"""Continue posture model_500 for 200 gait updates, saving every 50."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from train_lz_q1l_v1_support_500 import differences, write_json
from train_lz_q1l_v2_unified import ROOT, install_logging

TASK = 'lz_q1l_flat_v2_gait'
SOURCE = ROOT / 'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_posture/posture_resume20_to500_20261024_234507_3087344/model_500.pt'
SOURCE_SHA256 = 'ed10805f5778f24b1c249dbe254460473acb8ac39d85c9f85142fc2b93816e1f'
REVIEW = ROOT / 'media/v2_gait_reward_review_20261025/gpu_attempt01'


def preflight():
    """Check frozen-policy evidence and allow only the approved schedule edit."""
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    manifest = json.loads((REVIEW / 'manifest.json').read_text())
    assert all(json.loads((REVIEW / 'checks.json').read_text()).values())
    for name, digest in manifest['source_sha256'].items():
        path = Path(name)
        if path.name == 'lz_q1l_2_config.py':
            before = (REVIEW / 'provenance' / path.name).read_text()
            expected = before.replace(
                "experiment_name = 'flat_lz_q1l_v2_gait'\n        max_iterations = 100\n        save_interval = 10",
                "experiment_name = 'flat_lz_q1l_v2_gait'\n        max_iterations = 200  # New gait updates after posture model_500.\n        save_interval = 50")
            assert expected != before and path.read_text() == expected
        else:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    evidence = preflight()
    if args.preflight:
        print('GAIT_PREFLIGHT_OK source=posture500 updates=200 save_interval=50', flush=True)
        return
    if not args.gpu or not args.run_dir:
        parser.error('--gpu and --run-dir required')
    out = args.run_dir.resolve()
    logroot = ROOT / 'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_gait'
    if logroot.resolve() not in out.parents or out.exists():
        parser.error('Use a fresh run directory under logs2/flat_lz_q1l_v2_gait')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import isaacgym  # Must precede torch.
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import HIMOnPolicyRunner

    assert torch.cuda.is_available()
    cfg, train = task_registry.get_cfgs(TASK)
    assert train.runner.max_iterations == 200 and train.runner.save_interval == 50
    assert cfg.env.num_envs == 4096 and train.runner.num_steps_per_env == 100
    parent_cfg = json.loads((SOURCE.parent / 'env_config.json').read_text())
    parent_cfg['domain_rand'].pop('push_interval', None)
    env_diff = differences(parent_cfg, class_to_dict(cfg))
    assert env_diff == evidence['env_diff'], env_diff
    parent_train = json.loads((SOURCE.parent / 'train_config.json').read_text())
    train_diff = differences(parent_train, class_to_dict(train))
    assert set(train_diff) <= {'runner.experiment_name', 'runner.max_iterations', 'runner.save_interval'}, train_diff
    sys.argv = [sys.argv[0], '--task', TASK, '--headless', '--seed', str(train.seed),
                '--sim_device', 'cuda:0', '--rl_device', 'cuda:0']
    sim_args = get_args()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'train.log').symlink_to(str(out) + '.log')
    env = runner = None
    try:
        write_json(out / 'status.json', dict(state='initializing', completed_updates=0, target_updates=200))
        env, _ = task_registry.make_env(TASK, args=sim_args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train), str(out), device=sim_args.rl_device)
        runner.load(str(SOURCE), load_optimizer=True)
        assert runner.current_learning_iteration == 500
        assert runner.alg.optimizer.state and runner.alg.actor_critic.estimator.optimizer.state
        assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.state_dict().values())
        rates = {g['lr'] for g in runner.alg.optimizer.param_groups}
        assert len(rates) == 1
        runner.alg.learning_rate = rates.pop()
        # Local checkpoint labels count new gait updates; retain source count in metadata.
        runner.current_learning_iteration = 0
        env.take_diagnostics()
        metadata = dict(task=TASK, source_checkpoint=str(SOURCE), source_sha256=SOURCE_SHA256,
            source_iteration=500, start_iteration=0, target_iterations=200, additional_iterations=200,
            initialization='Full weights, both optimizers and saved PPO learning rate; new stage counter and restarted simulation/RNG',
            initial_learning_rate=runner.alg.learning_rate, save_interval=50,
            checkpoint_iterations=[50, 100, 150, 200], auto_stop=False,
            monitoring='Training rewards and direction/contact/force diagnostics every update; log only',
            gpu=args.gpu, pid=os.getpid(), seed=train.seed, reward_validation=str(REVIEW),
            feet=[env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])[i] for i in env.feet_indices.tolist()])
        sources = [ROOT / p for p in json.loads((SOURCE.parent / 'run_metadata.json').read_text())['code_sha256']]
        sources += [Path(__file__).resolve(), ROOT / 'tools/start_lz_q1l_v2_gait_200.sh',
                    ROOT / 'tools/queue_lz_q1l_v2_gait.py', ROOT / 'tools/check_lz_q1l_v2_gait.py']
        metadata['code_sha256'] = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = out / 'provenance' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            metadata['code_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        for name, value in [('run_metadata', metadata), ('env_config', class_to_dict(cfg)),
                            ('train_config', class_to_dict(train)),
                            ('config_diff_from_posture', dict(env=env_diff, training=train_diff))]:
            write_json(out / (name + '.json'), value)
        print('INITIALIZATION_OK', json.dumps(metadata), flush=True)
        install_logging(runner, out, metadata, 200)
        runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 200
        for step in metadata['checkpoint_iterations']:
            state = torch.load(str(out / ('model_%d.pt' % step)), map_location='cpu')
            assert state['iter'] == step
            assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        write_json(out / 'status.json', dict(state='completed', completed_updates=200, target_updates=200))
    except BaseException as exc:
        status = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
        status.update(state='stopped_by_user' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        write_json(out / 'status.json', status)
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
