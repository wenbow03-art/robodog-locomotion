"""Dedicated V4 runner; legacy runner and its archived behavior stay intact."""
import hashlib
import json
from pathlib import Path
import torch
from .him_on_policy_runner import HIMOnPolicyRunner
from rsl_rl.modules.him_lateral import HIMLateralActorCritic
from rsl_rl.algorithms.him_lateral_ppo import HIMLateralPPO


class HIMLateralRunner(HIMOnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        self.cfg=train_cfg['runner'];self.alg_cfg=train_cfg['algorithm'];self.policy_cfg=train_cfg['policy']
        self.device=device;self.env=env
        self.num_actor_obs=env.num_obs;self.num_critic_obs=env.num_privileged_obs or env.num_obs
        model=HIMLateralActorCritic(env.num_obs,self.num_critic_obs,env.num_one_step_obs,env.num_actions,**self.policy_cfg).to(device)
        model.configure_lateral(env.dof_names,env.commands_scale)
        from legged_gym import LEGGED_GYM_ROOT_DIR
        baseline=Path(LEGGED_GYM_ROOT_DIR)/self.cfg['baseline_directory']
        manifest=json.loads((baseline/'manifest.json').read_text())
        checkpoint=baseline/'model_800.pt'
        expected='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
        if manifest['checkpoints']['model_800.pt']['sha256']!=expected or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected:
            raise ValueError('Pinned new baseline model800 digest mismatch')
        saved=torch.load(checkpoint,map_location=device)
        if saved['iter']!=800:raise ValueError('Expected cumulative model800')
        model.load_new_baseline(saved['model_state_dict'])
        self.alg=HIMLateralPPO(model,device=device,**self.alg_cfg)
        self.num_steps_per_env=self.cfg['num_steps_per_env'];self.save_interval=self.cfg['save_interval']
        self.alg.init_storage(env.num_envs,self.num_steps_per_env,[env.num_obs],[self.num_critic_obs],[env.num_actions])
        self.log_dir=log_dir;self.writer=None;self.tot_timesteps=0;self.tot_time=0;self.current_learning_iteration=0
        self.alg.rollout_policy=env.build_inference_policy(model)
        env.reset()

    def get_inference_policy(self,device=None):
        if device is not None and torch.device(device)!=torch.device(self.device):
            raise ValueError('Create a V4 runner on the intended device to preserve live control state')
        self.alg.actor_critic.eval()
        return self.alg.rollout_policy

    def log(self,locs,width=80,pad=35):
        super().log(locs,width,pad)
        for key,value in getattr(self.alg,'metrics',{}).items():
            self.writer.add_scalar('V4/'+key,value,locs['it']+1)
