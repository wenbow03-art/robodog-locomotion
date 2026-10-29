"""Pinned inference-only experts; no optimizer or trainable composite checkpoint."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from rsl_rl.modules.him_actor_critic import HIMActorCritic
from rsl_rl.modules.him_right_rear_only import HIMRightRearOnlyActorCritic


class HIMLegacyLateralRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        from legged_gym import LEGGED_GYM_ROOT_DIR
        from legged_gym.utils.lz_q1l_legacy_lateral import LegacyLateralPolicy
        root = Path(LEGGED_GYM_ROOT_DIR)
        cfg = train_cfg['runner']
        if abs(env.cfg.sim.dt - .001) > 1e-9 or env.cfg.control.decimation != 20:
            raise ValueError('Composite requires baseline 1ms physics / 50Hz policy')
        self.device = torch.device(device)
        dims = (env.num_obs, env.num_privileged_obs, env.num_one_step_obs, env.num_actions)
        self.sources = {}
        def load(name, relative, sha, iteration, cls, policy_cfg):
            path = root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError('Checkpoint hash mismatch: ' + str(path))
            saved = torch.load(path, map_location=self.device)
            if saved['iter'] != iteration:
                raise ValueError('Unexpected checkpoint iteration')
            model = cls(*dims, **policy_cfg).to(self.device)
            model.load_state_dict(saved['model_state_dict'], strict=True)
            model.eval().requires_grad_(False)
            self.sources[name] = dict(path=str(path), sha256=sha, iteration=iteration)
            return model
        base = load('baseline', cfg['baseline_checkpoint'], cfg['baseline_sha256'], 800,
                    HIMRightRearOnlyActorCritic, train_cfg['policy'])
        # Validate saved named-joint mapping before allowing configuration.
        ids = [env.dof_names.index(k+'_leg_b_r_joint') for k in ('roll','pitch01','pitch02')]
        if base.right_rear_indices.tolist() != ids or not torch.equal(base.command_scale, env.commands_scale):
            raise ValueError('Baseline joint mapping or command scale mismatch')
        base.configure_right_rear(env.dof_names, env.commands_scale)
        old_cfg = json.loads((root / cfg['lateral_checkpoint']).with_name('train_config.json').read_text())
        lateral = load('lateral', cfg['lateral_checkpoint'], cfg['lateral_sha256'], 200,
                       HIMActorCritic, old_cfg['policy'])
        self.alg = SimpleNamespace(actor_critic=base)
        self.policy = LegacyLateralPolicy(base, lateral, env)
        env._stand_walk_actor = self.policy

    def get_inference_policy(self, device=None):
        if device is not None and torch.device(device) != self.device:
            raise ValueError('Construct runner on the target device')
        return self.policy

    def learn(self, *args, **kwargs):
        raise RuntimeError('Inference-only composite; training is not supported')

    def load(self, *args, **kwargs):
        raise RuntimeError('This composite loads its two hash-pinned sources automatically')
