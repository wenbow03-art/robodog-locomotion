"""Inference-only blend of frozen baseline and learned RH correction."""
import math
import torch


class StandWalkResidualActor:
    def __init__(self, actor, dt, blend_time_s=0.3, stand_threshold=0.05,
                 forward_threshold=0.1, other_command_tolerance=0.05):
        if not all(hasattr(actor, name) for name in
                   ('right_rear_delta', 'right_rear_indices', 'command_scale')):
            raise TypeError('Stand/walk switching requires an RH-only checkpoint')
        if not math.isfinite(blend_time_s) or blend_time_s <= 0 or dt <= 0:
            raise ValueError('Positive finite blend duration and control dt required')
        self.model = actor
        self.estimator = actor.estimator
        self.blend_step = min(float(dt)/blend_time_s, 1.)
        self.stand_threshold = stand_threshold
        self.forward_threshold = forward_threshold
        self.other_command_tolerance = other_command_tolerance
        self.alpha = None

    def reset(self, ids=None):
        if self.alpha is not None:
            if ids is None:
                self.alpha.zero_()
            else:
                self.alpha[ids] = 0.

    @torch.no_grad()
    def act_inference(self, history):
        model = self.model
        if self.alpha is None or len(self.alpha) != len(history):
            self.alpha = history.new_zeros(len(history))
        command = history[:, :3]/model.command_scale
        standing = (command.abs() <= self.stand_threshold).all(dim=1)
        forward = ((command[:, 0] >= self.forward_threshold)
                   & (command[:, 1:].abs() <= self.other_command_tolerance).all(dim=1))
        target = forward.to(history.dtype)
        self.alpha.add_((target-self.alpha).clamp(-self.blend_step, self.blend_step))
        self.alpha[self.alpha >= 1.-1e-6] = 1.
        self.alpha[self.alpha <= 1e-6] = 0.
        # Non-stand/non-forward commands retain the original baseline behavior.
        self.alpha[~(standing | forward)] = 0.
        velocity, latent = model.estimator(history)
        features = torch.cat((history[:, :model.num_one_step_obs], velocity, latent), dim=-1)
        baseline = model.actor(features)
        delta = model.right_rear_delta(features)
        result = baseline.clone()
        result[:, model.right_rear_indices] += self.alpha[:, None]*delta
        return result
