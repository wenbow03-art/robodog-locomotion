"""Fine-tune V7 for 200 updates, saving after exactly 50/100/150/200."""
import hashlib
import json
import os
from pathlib import Path

import isaacgym  # noqa: F401; must precede torch
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict


def main():
    args = get_args()
    assert args.task == 'lz_q1l_flat_v7' and args.max_iterations == 200
    args.headless = True
    args.resume = False
    source = Path(__file__).resolve().parents[1] / 'HIMLoco/legged_gym/logs1/flat_lz_q1l_v2/Oct04_21-48-35_v2_ft_v1_1700/model_200.pt'
    state = torch.load(str(source), map_location='cpu')
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    train_cfg.runner.save_interval = 50
    env, cfg = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, train_cfg = task_registry.make_alg_runner(env, args.task, args=args, train_cfg=train_cfg)
    runner.alg.actor_critic.load_state_dict(state['model_state_dict'])
    out = Path(runner.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(task=args.task, source_checkpoint=str(source),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    additional_iterations=200, save_interval=50, load_optimizer=False,
                    checkpoint_iterations=[50, 100, 150, 200],
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), pid=os.getpid())
    for name, data in [('run_metadata', metadata), ('env_config', class_to_dict(cfg)),
                       ('train_config', class_to_dict(train_cfg))]:
        (out / (name + '.json')).write_text(json.dumps(data, indent=2))
    print('RUN_CONFIG ' + json.dumps(metadata), flush=True)
    print('RUN_DIR ' + str(out), flush=True)

    # The upstream loop saves at zero-based indices and stores a stale counter.
    # Save at completed-update boundaries without changing the learning loop.
    original_log, original_save = runner.log, runner.save
    def log_and_checkpoint(locs):
        original_log(locs)
        completed = locs['it'] + 1
        (out / 'status.json').write_text(json.dumps({'state': 'running', 'completed_updates': completed}))
        if completed % 50 == 0:
            previous = runner.current_learning_iteration
            runner.current_learning_iteration = completed
            path = out / ('model_%d.pt' % completed)
            try:
                original_save(str(path) + '.tmp', infos=metadata)
                os.replace(str(path) + '.tmp', str(path))
            finally:
                runner.current_learning_iteration = previous
            print('CHECKPOINT_SAVED ' + str(path), flush=True)
    runner.log = log_and_checkpoint
    runner.save = lambda path, infos=None: None
    try:
        runner.learn(num_learning_iterations=200, init_at_random_ep_len=True)
        assert runner.current_learning_iteration == 200
        for step in (50, 100, 150, 200):
            assert (out / ('model_%d.pt' % step)).is_file()
    except BaseException as exc:
        (out / 'status.json').write_text(json.dumps({'state': 'failed', 'error': repr(exc)}))
        raise
    else:
        runner.writer.flush()
        (out / 'status.json').write_text(json.dumps({'state': 'completed', 'completed_updates': 200}))
        print('TRAINING_COMPLETE 200 updates', flush=True)


if __name__ == '__main__':
    main()
