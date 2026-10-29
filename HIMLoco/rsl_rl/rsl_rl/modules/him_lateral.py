"""Frozen stand/walk800 controller plus a direct, signed-command side residual."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from .him_right_rear_only import HIMRightRearOnlyActorCritic


class HIMLateralActorCritic(HIMRightRearOnlyActorCritic):
    def __init__(self, *args, lateral_hidden_dims=(256, 128), lateral_noise_std=.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.right_rear_delta.requires_grad_(False)
        layers=[]; width=self.num_actor_obs+19
        for hidden in lateral_hidden_dims:
            layers.extend([nn.Linear(width,hidden),nn.ELU()]);width=hidden
        layers.append(nn.Linear(width,12))
        self.lateral_delta=nn.Sequential(*layers)
        nn.init.zeros_(self.lateral_delta[-1].weight)
        nn.init.zeros_(self.lateral_delta[-1].bias)
        self.lateral_log_std=nn.Parameter(torch.full((12,),math.log(lateral_noise_std)))

    def configure_lateral(self, dof_names, command_scale):
        self.configure_right_rear(dof_names,command_scale)

    def load_new_baseline(self, state):
        expected={k for k in self.state_dict() if not k.startswith('lateral_delta.') and k!='lateral_log_std'}
        if set(state)!=expected:
            raise ValueError('V4 must initialize from the full new baseline model800')
        self.load_state_dict(state,strict=False)
        nn.init.zeros_(self.lateral_delta[-1].weight)
        nn.init.zeros_(self.lateral_delta[-1].bias)

    def lateral_mask(self, history):
        c=history[:,:3]/self.command_scale
        # Pure lateral commands only; no diagonal scope or reflection.
        return (c[:,1].abs()>.1)&(c[:,[0,2]].abs()<=1e-6).all(dim=1)

    def lateral_actions(self, history):
        if not self._joints_configured:raise RuntimeError('Configure named joints before inference')
        with torch.no_grad():
            v,z=self.estimator(history)
            base=self.actor(torch.cat((history[:,:self.num_one_step_obs],v,z),dim=-1))
        # Raw six-frame history lets the new branch learn lateral temporal features
        # without changing the frozen stand/walk estimator.
        delta=self.lateral_delta(torch.cat((history,v,z),dim=-1))
        return base+delta

    def update_distribution(self, history, mean=None):
        if mean is None:mean=self.lateral_actions(history)
        self.distribution=Normal(mean,self.lateral_log_std.exp().expand_as(mean))

    def optimization_parameters(self):
        return list(self.lateral_delta.parameters())+[self.lateral_log_std]+list(self.critic.parameters())

    def frozen_policy_parameters(self):
        return super().frozen_policy_parameters()+list(self.right_rear_delta.parameters())

    def train(self,mode=True):
        super().train(mode)
        self.right_rear_delta.eval()
        return self
