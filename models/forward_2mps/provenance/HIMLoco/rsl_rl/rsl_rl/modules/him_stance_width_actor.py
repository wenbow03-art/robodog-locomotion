"""Keep model800 outputs exactly at initialization, allow all four legs to learn."""
import torch
from .him_right_rear_only import HIMRightRearOnlyActorCritic


class HIMStanceWidthActorCritic(HIMRightRearOnlyActorCritic):
    def __init__(self, *args, continuous_rh_forward=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor.requires_grad_(True)
        self.continuous_rh_forward = continuous_rh_forward

    def correction_mask(self, obs_history):
        original = super().correction_mask(obs_history)
        if not self.continuous_rh_forward:
            return original
        cmd = obs_history[:, :3] / self.command_scale
        gap = (cmd[:, 0] > .05) & (cmd[:, 0] < .1) & (cmd[:, 1:].abs() <= .05).all(1)
        return original | gap

    def train(self, mode=True):
        super().train(mode)
        self.actor.train(mode)
        return self

    def act_inference(self, obs_history, observations=None):
        if not self._joints_configured:
            raise RuntimeError('Configure named joints first')
        with torch.no_grad():
            vel, latent = self.estimator(obs_history)
            features = torch.cat((obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1)
        baseline = self.actor(features)
        delta = self.right_rear_delta(features)
        delta = delta * self.correction_mask(obs_history).unsqueeze(1)
        result = baseline.clone()
        result[:, self.right_rear_indices] = baseline[:, self.right_rear_indices] + delta
        return result

    def frozen_policy_parameters(self):
        return list(self.estimator.parameters()) + [self.std]

    def optimization_parameters(self):
        # Preserve old optimizer ordering, append newly trainable actor parameters.
        return super().optimization_parameters() + list(self.actor.parameters())
