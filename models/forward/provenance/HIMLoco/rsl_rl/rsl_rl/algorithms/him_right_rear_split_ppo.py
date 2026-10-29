"""RH-only PPO with separate standing/forward teacher weights."""
import math
import torch
from .him_right_rear_only_ppo import HIMRightRearOnlyPPO


class HIMRightRearSplitPPO(HIMRightRearOnlyPPO):
    def __init__(self, actor_critic, rear_teacher_forward_coef=0.3, **kwargs):
        if not math.isfinite(rear_teacher_forward_coef) or rear_teacher_forward_coef < 0:
            raise ValueError('Invalid forward teacher coefficient')
        super().__init__(actor_critic, **kwargs)
        self.rear_teacher_forward_coef = rear_teacher_forward_coef
        self._split_totals = [0., 0., 0, 0]

    def distillation_loss(self, observations, action_mean):
        ids = self.command_mask(observations).nonzero(as_tuple=False).flatten()
        if len(ids) > self.teacher_max_samples:
            ids = ids[torch.randperm(len(ids), device=ids.device)[:self.teacher_max_samples]]
        if len(ids) == 0:
            return action_mean.sum()*0., dict(rear_mse=0., other_mse=0., weighted_loss=0., samples=0)
        rear_target, other_target = self.teacher_targets(observations[ids])
        if not torch.isfinite(rear_target).all() or not torch.isfinite(other_target).all():
            raise RuntimeError('Nonfinite teacher target')
        mean = action_mean[ids]
        errors = (mean.index_select(1, self.rear_indices)-rear_target).square().mean(dim=1)
        commands = observations[ids, :3]/self.command_scale
        stand = (commands.abs() <= self.teacher_stand_threshold).all(dim=1)
        weights = torch.where(stand, errors.new_tensor(self.rear_teacher_coef),
                              errors.new_tensor(self.rear_teacher_forward_coef))
        # Preserve the previous reduction: mean across all eligible samples.
        # Equal weights reproduce the old loss; only forward samples get 3x.
        loss = (weights*errors).mean()
        other_mse = (mean.index_select(1, self.other_indices)-other_target).square().mean()
        with torch.no_grad():
            self._split_totals[0] += errors[stand].sum().item()
            self._split_totals[1] += errors[~stand].sum().item()
            self._split_totals[2] += int(stand.sum())
            self._split_totals[3] += int((~stand).sum())
        return loss, dict(rear_mse=errors.mean().item(), other_mse=other_mse.item(),
                          weighted_loss=loss.item(), samples=len(ids))

    def update(self):
        self._split_totals = [0., 0., 0, 0]
        result = super().update()
        stand_sum, forward_sum, stand_n, forward_n = self._split_totals
        self.distillation_metrics.update(
            stand_mse=stand_sum/max(stand_n, 1), forward_mse=forward_sum/max(forward_n, 1),
            stand_samples=stand_n, forward_samples=forward_n,
            stand_coefficient=self.rear_teacher_coef,
            forward_coefficient=self.rear_teacher_forward_coef)
        return result
