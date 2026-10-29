"""PPO with a frozen estimator, without a teacher or imitation objective."""
import torch
from torch import nn
from .him_ppo import HIMPPO


class HIMFrozenEstimatorPPO(HIMPPO):
    def __init__(self, actor_critic, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.actor_critic.estimator.requires_grad_(False)
        self.actor_critic.estimator.eval()
        self.parameters=[p for p in actor_critic.parameters() if p.requires_grad]
        self.optimizer=torch.optim.Adam(self.parameters,lr=self.learning_rate)
    def update(self):
        model=self.actor_critic;totals=[0.,0.,0.];batches=0
        for obs,crit,actions,next_crit,old_value,adv,returns,old_logp,old_mu,old_sigma in self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs):
            model.update_distribution(obs)
            logp=model.get_actions_log_prob(actions)
            value=model.evaluate(crit)
            ratio=(logp-old_logp.flatten()).exp()
            surrogate=torch.maximum(-adv.flatten()*ratio,-adv.flatten()*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
            clipped=old_value+(value-old_value).clamp(-self.clip_param,self.clip_param)
            vloss=torch.maximum((value-returns).square(),(clipped-returns).square()).mean() if self.use_clipped_value_loss else (value-returns).square().mean()
            loss=surrogate+self.value_loss_coef*vloss-self.entropy_coef*model.entropy.mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite frozen-estimator PPO loss')
            self.optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.parameters,self.max_grad_norm);self.optimizer.step()
            with torch.no_grad():model.std.clamp_(.05,1.)
            with torch.no_grad():
                kl=(torch.log(model.action_std/old_sigma)+(old_sigma.square()+(old_mu-model.action_mean).square())/(2*model.action_std.square())-.5).sum(-1).mean()
            for i,x in enumerate((vloss,surrogate,kl)):totals[i]+=float(x)
            batches+=1
        self.storage.clear()
        self.metrics=dict(kl=totals[2]/batches)
        return totals[0]/batches,totals[1]/batches,0.,0.
