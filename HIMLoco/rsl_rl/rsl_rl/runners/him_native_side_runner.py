"""Original V8 initialization, native side-only anchored PPO."""
import hashlib
from pathlib import Path
import torch
from .him_on_policy_runner import HIMOnPolicyRunner
from rsl_rl.algorithms.him_anchored_ppo import HIMAnchoredPPO


class HIMNativeSideRunner(HIMOnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        super().__init__(env,train_cfg,log_dir,device)
        from legged_gym import LEGGED_GYM_ROOT_DIR
        path=Path(LEGGED_GYM_ROOT_DIR)/'logs1/flat_lz_q1l_v8/v8_ft_v7_200_200iters_20261018_130426_2712609/model_200.pt'
        if hashlib.sha256(path.read_bytes()).hexdigest()!='3480ca2abddfcd62df2be30d89e53a992e3eadc100add59df131213feef48a5a':raise ValueError('Original V8 hash mismatch')
        model=self.alg.actor_critic
        model.load_state_dict(torch.load(path,map_location=device)['model_state_dict'],strict=True)
        del self.alg
        self.alg=HIMAnchoredPPO(model,anchor_coef=train_cfg['anchor']['coefficient'],device=device,**self.alg_cfg)
        self.alg.install_teacher()
        self.alg.init_storage(env.num_envs,self.num_steps_per_env,[env.num_obs],[env.num_privileged_obs],[env.num_actions])

    def log(self,locs,width=80,pad=35):
        super().log(locs,width,pad)
        for key,value in getattr(self.alg,'metrics',{}).items():
            self.writer.add_scalar('Anchor/'+key,value,locs['it']+1)
