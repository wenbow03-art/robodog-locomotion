"""Moving LH teacher; PPO trains all legs, imitation trains the RH correction only."""
import torch
from .him_right_rear_split_ppo import HIMRightRearSplitPPO
from rsl_rl.modules.him_stance_width_actor import HIMStanceWidthActorCritic


class HIMEmaRearTeacherPPO(HIMRightRearSplitPPO):
    def __init__(self, actor_critic, teacher_ema_tau=.05, **kwargs):
        if not isinstance(actor_critic,HIMStanceWidthActorCritic):
            raise TypeError('Moving teacher requires the trainable all-leg actor')
        if not 0. < teacher_ema_tau <= 1.:
            raise ValueError('EMA tau must be in (0,1]')
        self.teacher_ema_tau=teacher_ema_tau
        self.teacher_updates=0
        super().__init__(actor_critic,**kwargs)

    def distillation_loss(self, observations, action_mean):
        model=self.actor_critic
        # Detach the shared actor on this loss path, not on the PPO path.
        # Thus imitation cannot drag LH/front outputs through shared parameters.
        with torch.no_grad():
            vel,latent=model.estimator(observations)
            features=torch.cat((observations[:,:model.num_one_step_obs],vel,latent),dim=-1)
            baseline=model.actor(features)
        delta=model.right_rear_delta(features)*model.correction_mask(observations).unsqueeze(1)
        mean=baseline.clone()
        mean[:,model.right_rear_indices]=baseline[:,model.right_rear_indices]+delta
        return super().distillation_loss(observations,mean)

    @torch.no_grad()
    def update_teacher(self):
        if self.teacher is None:raise RuntimeError('Attach EMA teacher before updating')
        target=dict(self.teacher.named_parameters())
        for name,source in self.actor_critic.named_parameters():
            if source.requires_grad:
                target[name].mul_(1.-self.teacher_ema_tau).add_(source,alpha=self.teacher_ema_tau)
            else:
                target[name].copy_(source)
        for name,source in self.actor_critic.named_buffers():
            dict(self.teacher.named_buffers())[name].copy_(source)
        self.teacher_updates+=1

    def update(self):
        result=super().update()
        self.update_teacher()
        self.distillation_metrics.update(teacher_updates=self.teacher_updates,teacher_ema_tau=self.teacher_ema_tau)
        return result
