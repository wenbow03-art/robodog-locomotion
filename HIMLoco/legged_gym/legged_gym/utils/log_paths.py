"""Storage roots after the user's logs -> logs1/logs2 reorganization."""
import os
from legged_gym import LEGGED_GYM_ROOT_DIR


def get_experiment_log_dir(train_cfg):
    directory = getattr(train_cfg.runner, 'log_directory', 'logs1')
    return os.path.join(LEGGED_GYM_ROOT_DIR, directory, train_cfg.runner.experiment_name)
