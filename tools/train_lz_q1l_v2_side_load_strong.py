"""Continue gait model_150 for 300 one-sided strong load updates, saving every 10."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from train_lz_q1l_v1_support_500 import differences, write_json
from train_lz_q1l_v2_unified import ROOT, install_logging

TASK = 'lz_q1l_flat_v2_side_load_strong'
SOURCE = ROOT / 'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_gait/gait_from_posture500_200_20261025_113614_3235946/model_150.pt'
SOURCE_SHA256 = '655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747'
REVIEW = ROOT / 'media/side_load_strong_validation_20261026'


EXPECTED_ADDITIONS = {
    'rewards.side_force_window_s': 1.,
    'rewards.side_force_epsilon_n': 1.,
    'rewards.scales.left_load_excess': -10.,
}


def preflight():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    parent = json.loads((SOURCE.parent / 'run_metadata.json').read_text())
    for name, digest in parent['code_sha256'].items():
        if name.endswith('/envs/__init__.py'):
            continue
        if name.endswith(('/lz_q1l_2.py', '/lz_q1l_2_config.py')):
            archived = SOURCE.parent / 'provenance' / name
            assert hashlib.sha256(archived.read_bytes()).hexdigest() == digest
            assert (ROOT/name).read_bytes().startswith(archived.read_bytes()), name
        else:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    validation = json.loads((REVIEW/'report.json').read_text())
    assert validation['finite'] and validation['checkpoint_sha256'] == SOURCE_SHA256
    assert validation['training_updates'] == 0 and validation['checkpoint_unchanged']
    assert validation['optimizer_restore_checked']
    assert all(v for k,v in validation['checks'].items() if k!='parent_gait_checks')
    assert all(validation['checks']['parent_gait_checks'].values())
    for name,digest in validation['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    return dict(config_diff={k:dict(v1=None,this_run=v) for k,v in EXPECTED_ADDITIONS.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    evidence = preflight()
    if args.preflight:
        print('SIDE_LOAD_STRONG_PREFLIGHT_OK source=gait150 updates=300 save_interval=10', flush=True)
        return
    if not args.gpu or not args.run_dir:
        parser.error('--gpu and --run-dir required')
    out = args.run_dir.resolve()
    logroot = ROOT / 'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_side_load_strong'
    if logroot.resolve() not in out.parents or out.exists():
        parser.error('Use a fresh run directory under logs2/flat_lz_q1l_v2_side_load_strong')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import isaacgym  # Must precede torch.
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import HIMOnPolicyRunner

    assert torch.cuda.is_available()
    cfg, train = task_registry.get_cfgs(TASK)
    assert train.runner.max_iterations == 300 and train.runner.save_interval == 10
    assert cfg.env.num_envs == 4096 and train.runner.num_steps_per_env == 100
    parent_cfg = json.loads((SOURCE.parent / 'env_config.json').read_text())
    parent_cfg['domain_rand'].pop('push_interval', None)
    env_diff = differences(parent_cfg, class_to_dict(cfg))
    assert env_diff == evidence['config_diff'], env_diff
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
        write_json(out / 'status.json', dict(state='initializing', completed_updates=0, target_updates=300))
        env, _ = task_registry.make_env(TASK, args=sim_args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train), str(out), device=sim_args.rl_device)
        runner.load(str(SOURCE), load_optimizer=True)
        assert runner.current_learning_iteration == 150
        assert runner.alg.optimizer.state and runner.alg.actor_critic.estimator.optimizer.state
        assert all(torch.isfinite(v).all() for v in runner.alg.actor_critic.state_dict().values())
        rates = {g['lr'] for g in runner.alg.optimizer.param_groups}
        assert len(rates) == 1
        runner.alg.learning_rate = rates.pop()
        # Local checkpoint labels count new gait updates; retain source count in metadata.
        runner.current_learning_iteration = 0
        env.take_diagnostics()
        metadata = dict(task=TASK, source_checkpoint=str(SOURCE), source_sha256=SOURCE_SHA256,
            source_iteration=150, start_iteration=0, target_iterations=300, additional_iterations=300,
            initialization='Full weights, both optimizers and saved PPO learning rate; new stage counter and restarted simulation/RNG',
            initial_learning_rate=runner.alg.learning_rate, save_interval=10,
            checkpoint_iterations=list(range(10,301,10)), auto_stop=False,
            monitoring='Training rewards and direction/contact/force diagnostics every update; log only',
            gpu=args.gpu, pid=os.getpid(), seed=train.seed, reward_validation=str(REVIEW),
            feet=[env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])[i] for i in env.feet_indices.tolist()])
        sources = [ROOT / p for p in json.loads((SOURCE.parent / 'run_metadata.json').read_text())['code_sha256']]
        sources += [Path(__file__).resolve(), ROOT / 'tools/start_lz_q1l_v2_side_load_strong_300.sh',
                    ROOT / 'tools/queue_lz_q1l_v2_side_load_strong.py', ROOT / 'tools/check_lz_q1l_side_load_strong.py',
                    ROOT / 'tools/eval_lz_q1l_side_load_strong.py', ROOT / 'tools/audit_lz_q1l_new_lineage.py',
                    ROOT / 'tools/summarize_lz_q1l_lineage_audit.py']
        metadata['code_sha256'] = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            target = out / 'provenance' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            metadata['code_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        for name, value in [('run_metadata', metadata), ('env_config', class_to_dict(cfg)),
                            ('train_config', class_to_dict(train)),
                            ('config_diff_from_gait150', dict(env=env_diff, training=train_diff))]:
            write_json(out / (name + '.json'), value)
        print('INITIALIZATION_OK', json.dumps(metadata), flush=True)
        install_logging(runner, out, metadata, 300)
        runner.learn(num_learning_iterations=300, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 300
        for step in metadata['checkpoint_iterations']:
            state = torch.load(str(out / ('model_%d.pt' % step)), map_location='cpu')
            assert state['iter'] == step
            assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        write_json(out / 'status.json', dict(state='completed', completed_updates=300, target_updates=300))
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
