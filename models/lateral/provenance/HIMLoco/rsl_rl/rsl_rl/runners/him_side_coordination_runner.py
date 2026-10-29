"""Initialize coordination fine-tuning from the reviewed native 1 ms model300."""
import hashlib
from pathlib import Path
import torch
from .him_on_policy_runner import HIMOnPolicyRunner
from rsl_rl.algorithms.him_frozen_estimator_ppo import HIMFrozenEstimatorPPO


class HIMSideCoordinationRunner(HIMOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        super().__init__(env, train_cfg, log_dir, device)
        from legged_gym import LEGGED_GYM_ROOT_DIR
        path = Path(LEGGED_GYM_ROOT_DIR) / self.cfg['source_checkpoint']
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.cfg['source_sha256']:
            raise ValueError('Coordination source model300 hash mismatch')
        saved = torch.load(path, map_location=device)
        if saved['iter'] != 300:
            raise ValueError('Expected final 1 ms model300')
        model = self.alg.actor_critic
        model.load_state_dict(saved['model_state_dict'], strict=True)
        del self.alg
        self.alg = HIMFrozenEstimatorPPO(model, device=device, **self.alg_cfg)
        self.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        model.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
        if any(group['lr'] != self.alg.learning_rate for group in self.alg.optimizer.param_groups):
            raise ValueError('Source optimizer learning rate differs from config')
        self.alg.init_storage(env.num_envs, self.num_steps_per_env, [env.num_obs],
                              [env.num_privileged_obs], [env.num_actions])
        self.sources = dict(checkpoint=str(path), sha256=self.cfg['source_sha256'], source_iteration=300)
        self.current_learning_iteration = 0

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        for name, value in self.env.pop_side_coordination_metrics().items():
            self.writer.add_scalar('SideCoordination/' + name, value, locs['it'] + 1)
