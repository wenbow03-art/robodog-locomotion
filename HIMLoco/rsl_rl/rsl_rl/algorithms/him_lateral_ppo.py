"""PPO updates only lateral residual/noise and critic; baseline is immutable."""
import math
import torch
from torch import nn
from .him_ppo import HIMPPO


class HIMLateralPPO(HIMPPO):
    def __init__(self,actor_critic,**kwargs):
        super().__init__(actor_critic,**kwargs)
        if self.schedule!='fixed':raise ValueError('V4 initial review uses a fixed learning rate')
        self.optimizer=torch.optim.Adam(actor_critic.optimization_parameters(),lr=self.learning_rate)
        self.rollout_policy=None
        self.assert_frozen_policy()

    def assert_frozen_policy(self):
        frozen=self.actor_critic.frozen_policy_parameters()
        assert all(not p.requires_grad for p in frozen)
        actual={id(p) for g in self.optimizer.param_groups for p in g['params']}
        assert actual=={id(p) for p in self.actor_critic.optimization_parameters()}
        assert not actual.intersection(id(p) for p in frozen)

    def act(self,obs,critic_obs):
        if self.rollout_policy is None:raise RuntimeError('Install the environment inference factory first')
        model=self.actor_critic; mean=self.rollout_policy(obs)
        model.update_distribution(obs,mean=mean)
        lateral=model.lateral_mask(obs)
        # Standing/forward are deterministic new-baseline rehearsal, without noise.
        actions=torch.where(lateral[:,None],model.distribution.sample(),mean)
        self.transition.actions=actions.detach()
        self.transition.values=model.evaluate(critic_obs).detach()
        logp=model.get_actions_log_prob(actions)
        self.transition.actions_log_prob=torch.where(lateral,logp,torch.zeros_like(logp)).detach()
        self.transition.action_mean=mean.detach()
        self.transition.action_sigma=model.action_std.detach()
        self.transition.observations=obs
        self.transition.critic_observations=critic_obs
        return actions

    def update(self):
        self.assert_frozen_policy()
        model=self.actor_critic; totals=[0.,0.]; lateral_n=0;kl_sum=0.;batch_n=0
        generator=self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs)
        for obs,crit,actions,next_crit,old_value,adv,returns,old_logp,old_mu,old_std in generator:
            value=model.evaluate(crit)
            if self.use_clipped_value_loss:
                clipped=old_value+(value-old_value).clamp(-self.clip_param,self.clip_param)
                value_loss=torch.maximum((value-returns).square(),(clipped-returns).square()).mean()
            else:value_loss=(value-returns).square().mean()
            mask=model.lateral_mask(obs)
            surrogate=value.new_zeros(()); entropy=value.new_zeros(())
            if mask.any():
                # Stateful stand/walk blend NEVER advances during shuffled PPO updates.
                model.update_distribution(obs[mask])
                logp=model.get_actions_log_prob(actions[mask])
                a=adv[mask].flatten();a=(a-a.mean())/(a.std(unbiased=False)+1e-8)
                ratio=(logp-old_logp[mask].flatten()).exp()
                surrogate=torch.maximum(-a*ratio,-a*ratio.clamp(1-self.clip_param,1+self.clip_param)).mean()
                entropy=model.entropy.mean()
                with torch.no_grad():
                    mu,sigma=model.action_mean,model.action_std
                    kl=(torch.log(sigma/old_std[mask])+(old_std[mask].square()+(old_mu[mask]-mu).square())/(2*sigma.square())-.5).sum(-1).mean()
                    kl_sum+=float(kl);lateral_n+=int(mask.sum())
            loss=surrogate+self.value_loss_coef*value_loss-self.entropy_coef*entropy
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite V4 PPO loss')
            self.optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(model.optimization_parameters(),self.max_grad_norm)
            self.optimizer.step()
            with torch.no_grad():model.lateral_log_std.clamp_(math.log(.05),math.log(.5))
            totals[0]+=float(value_loss);totals[1]+=float(surrogate);batch_n+=1
        self.metrics=dict(lateral_samples=lateral_n,kl=kl_sum/max(batch_n,1),
                          lateral_std=float(model.lateral_log_std.exp().mean()))
        self.storage.clear()
        return totals[0]/batch_n,totals[1]/batch_n,0.,0.
