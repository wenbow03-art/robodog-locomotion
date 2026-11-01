# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

"""PPO with LH-to-RH supervision and a structurally frozen other-leg policy."""
import torch
import torch.nn as nn
from .him_rear_teacher_ppo import HIMRearTeacherPPO
from rsl_rl.modules.him_right_rear_only import HIMRightRearOnlyActorCritic


class HIMRightRearOnlyPPO(HIMRearTeacherPPO):
    def __init__(self, actor_critic, other_legs_coef=0., **kwargs):
        if not isinstance(actor_critic, HIMRightRearOnlyActorCritic):
            raise TypeError('RH-only PPO requires the isolated RH architecture')
        if other_legs_coef != 0.:
            raise ValueError('No soft other-leg penalty: those policies must be frozen')
        super().__init__(actor_critic, other_legs_coef=0., **kwargs)
        self.optimizer=torch.optim.Adam(actor_critic.optimization_parameters(), lr=self.learning_rate)
        self.assert_frozen_policy()

    def assert_frozen_policy(self):
        frozen=self.actor_critic.frozen_policy_parameters()
        if any(p.requires_grad for p in frozen):
            raise RuntimeError('Baseline policy/estimator/std must remain frozen')
        optimized={id(p) for group in self.optimizer.param_groups for p in group['params']}
        expected={id(p) for p in self.actor_critic.optimization_parameters()}
        if optimized != expected or optimized.intersection(id(p) for p in frozen):
            raise RuntimeError('Optimizer must contain only RH branch and critic')

    def set_teacher(self, teacher, mirror, dof_names, command_scale):
        super().set_teacher(teacher, mirror, dof_names, command_scale)
        if not torch.equal(self.rear_indices,self.actor_critic.right_rear_indices):
            raise RuntimeError('Teacher/student RH mapping mismatch')

    def update(self):
        # Only RH correction and critic may change; estimator is never updated.
        self.assert_frozen_policy()
        mean_value_loss = mean_surrogate_loss = mean_estimation_loss = mean_swap_loss = 0.
        totals = dict(rear_mse=0., other_mse=0., weighted_loss=0., samples=0.)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch in generator:
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(torch.log(sigma_batch / old_sigma_batch + 1.e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate
            estimation_loss, swap_loss = 0., 0.  # Frozen estimator: no update or prototype normalization.
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch-target_values_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch-returns_batch).pow(2)
                value_losses_clipped = (value_clipped-returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch-value_batch).pow(2).mean()
            supervision, stats = self.distillation_loss(obs_batch, mu_batch)
            loss = surrogate_loss + self.value_loss_coef*value_loss - self.entropy_coef*entropy_batch.mean() + supervision
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.optimization_parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss
            for key in totals:
                totals[key] += stats[key]
        count = self.num_learning_epochs*self.num_mini_batches
        self.distillation_metrics = {key: value/count for key,value in totals.items()}
        self.storage.clear()
        return mean_value_loss/count, mean_surrogate_loss/count, mean_estimation_loss/count, mean_swap_loss/count
