"""Native-direction PPO with a frozen old-policy action reference."""
import copy
import torch
from torch import nn
from .him_ppo import HIMPPO


class HIMAnchoredPPO(HIMPPO):
    def __init__(self, actor_critic, anchor_coef=1., **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.anchor_coef=anchor_coef
        self.actor_critic.estimator.requires_grad_(False)
        self.actor_critic.estimator.eval()
        self.parameters=[p for p in actor_critic.parameters() if p.requires_grad]
        self.optimizer=torch.optim.Adam(self.parameters,lr=self.learning_rate)
        self.teacher=None

    def install_teacher(self):
        self.teacher=copy.deepcopy(self.actor_critic).eval().requires_grad_(False)

    def anchor_loss(self, obs, mean):
        if self.teacher is None:raise RuntimeError('Install original source teacher before training')
        with torch.no_grad():target=self.teacher.act_inference(obs)
        # Same physical observations/direction, no mirror or joint permutation.
        scale=self.teacher.std.detach().abs().clamp_min(.1)
        return ((mean-target)/scale).square().mean()

    def update(self):
        model=self.actor_critic;totals=[0.,0.,0.,0.];batches=0
        for obs,crit,actions,next_crit,old_value,adv,returns,old_logp,old_mu,old_sigma in self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs):
            model.update_distribution(obs)
            logp=model.get_actions_log_prob(actions)
            value=model.evaluate(crit)
            ratio=(logp-old_logp.flatten()).exp()
            surrogate=torch.maximum(-adv.flatten()*ratio,-adv.flatten()*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
            clipped=old_value+(value-old_value).clamp(-self.clip_param,self.clip_param)
            vloss=torch.maximum((value-returns).square(),(clipped-returns).square()).mean() if self.use_clipped_value_loss else (value-returns).square().mean()
            anchor=self.anchor_loss(obs,model.action_mean)
            loss=surrogate+self.value_loss_coef*vloss-self.entropy_coef*model.entropy.mean()+self.anchor_coef*anchor
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite anchored PPO loss')
            self.optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(self.parameters,self.max_grad_norm);self.optimizer.step()
            with torch.no_grad():model.std.clamp_(.05,1.)
            with torch.no_grad():
                kl=(torch.log(model.action_std/old_sigma)+(old_sigma.square()+(old_mu-model.action_mean).square())/(2*model.action_std.square())-.5).sum(-1).mean()
            for i,x in enumerate((vloss,surrogate,anchor,kl)):totals[i]+=float(x)
            batches+=1
        self.storage.clear()
        self.metrics=dict(anchor_loss=totals[2]/batches,kl=totals[3]/batches)
        return totals[0]/batches,totals[1]/batches,0.,0.
