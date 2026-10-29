"""HIMLoco training entry for LZ-Q1L.

Same as legged_gym/scripts/train.py, plus:
    --init_from PATH   initialise the policy/estimator weights from a checkpoint (optimizer state is not loaded)
"""
import sys

import isaacgym  # noqa: F401  (must be imported before torch)
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    init_from = None
    if "--init_from" in sys.argv:
        i = sys.argv.index("--init_from")
        init_from = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    args = get_args()
    args.headless = True
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    if init_from is not None:
        state = torch.load(init_from, map_location=runner.device)
        runner.alg.actor_critic.load_state_dict(state["model_state_dict"])
        print("initialised weights from", init_from)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
