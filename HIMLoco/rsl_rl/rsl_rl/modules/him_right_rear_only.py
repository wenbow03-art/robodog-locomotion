"""Frozen baseline controller plus an independent THREE-output RH correction."""
import torch
from torch import nn
from torch.distributions import Normal

from .him_actor_critic import HIMActorCritic


class HIMRightRearOnlyActorCritic(HIMActorCritic):
    def __init__(self, *args, right_rear_hidden_dims=(128, 64), **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_actions != 12 or self.num_actor_obs != 270 or self.num_one_step_obs != 45:
            raise ValueError('RH-only controller requires the LZ-Q1L 12-action/6x45 layout')
        for module in (self.actor, self.estimator):
            module.requires_grad_(False)
            module.eval()
        # Preserve baseline exploration as well as its deterministic other-leg outputs.
        self.std.requires_grad_(False)
        layers=[]; width=self.num_one_step_obs+19
        for hidden in right_rear_hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ELU()]);width=hidden
        layers.append(nn.Linear(width, 3))
        self.right_rear_delta=nn.Sequential(*layers)
        nn.init.zeros_(self.right_rear_delta[-1].weight)
        nn.init.zeros_(self.right_rear_delta[-1].bias)
        self.register_buffer('right_rear_indices', torch.full((3,), -1, dtype=torch.long))
        self.register_buffer('command_scale', torch.ones(3))
        self._joints_configured=False

    def configure_right_rear(self, dof_names, command_scale):
        names=list(dof_names)
        expected={kind+'_leg_'+leg+'_joint' for kind in ('roll','pitch01','pitch02')
                  for leg in ('b_l','b_r','f_l','f_r')}
        if len(names)!=12 or set(names)!=expected:
            raise ValueError('Unexpected LZ-Q1L joint names')
        ids=[names.index(kind+'_leg_b_r_joint') for kind in ('roll','pitch01','pitch02')]
        scale=torch.as_tensor(command_scale,device=self.command_scale.device)
        if scale.shape!=(3,) or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError('Invalid command scale')
        self.right_rear_indices.copy_(torch.tensor(ids,device=self.right_rear_indices.device))
        self.command_scale.copy_(scale)
        self._joints_configured=True

    def load_baseline(self, state_dict):
        """Load old gait150 keys exactly; only the new zero branch/buffers are absent."""
        expected={k for k in self.state_dict() if not k.startswith('right_rear_delta.')
                  and k not in ('right_rear_indices','command_scale')}
        if set(state_dict)!=expected:
            raise ValueError('Baseline parameter keys do not match frozen architecture')
        result=self.load_state_dict(state_dict,strict=False)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
        # Loading baseline into a previously used instance must reset its correction.
        nn.init.zeros_(self.right_rear_delta[-1].weight)
        nn.init.zeros_(self.right_rear_delta[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.actor.eval();self.estimator.eval()
        return self

    def correction_mask(self, obs_history):
        commands=obs_history[:, :3]/self.command_scale
        stand=(commands.abs()<=.05).all(dim=1)
        forward=(commands[:,0]>=.10)&(commands[:,1:].abs()<=.05).all(dim=1)
        return stand|forward

    def act_inference(self, obs_history, observations=None):
        if not self._joints_configured:
            raise RuntimeError('Configure named RH joints before evaluating this controller')
        with torch.no_grad():
            vel,latent=self.estimator(obs_history)
            features=torch.cat((obs_history[:,:self.num_one_step_obs],vel,latent),dim=-1)
            baseline=self.actor(features)
        delta=self.right_rear_delta(features.detach())
        delta=delta*self.correction_mask(obs_history).unsqueeze(1)
        # Only these three columns can change. No learned weights feed the other nine.
        result=baseline.clone()
        result[:,self.right_rear_indices]=baseline[:,self.right_rear_indices]+delta
        return result

    def update_distribution(self, obs_history):
        mean=self.act_inference(obs_history)
        self.distribution=Normal(mean,mean*0.+self.std)

    def frozen_policy_parameters(self):
        return list(self.actor.parameters())+list(self.estimator.parameters())+[self.std]

    def optimization_parameters(self):
        return list(self.right_rear_delta.parameters())+list(self.critic.parameters())
