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

"""Optional new-v3 actor supervision; the original HIMPPO remains unchanged.

The frozen gait150 teacher sees mirrored histories. Only its mirrored-back
right-rear outputs supervise that leg. Other legs track the unmirrored teacher.
Teacher loading is explicit in the training entrypoint, not required for inference.
"""
import math

import torch
import torch.nn as nn

from .him_ppo import HIMPPO


class HIMRearTeacherPPO(HIMPPO):
    def __init__(self, actor_critic, rear_teacher_coef=0.1, other_legs_coef=0.02,
                 teacher_max_samples=4096, teacher_stand_threshold=0.05,
                 teacher_forward_threshold=0.1, teacher_other_command_tolerance=0.05,
                 **kwargs):
        super().__init__(actor_critic, **kwargs)
        for name, value in [('rear_teacher_coef', rear_teacher_coef), ('other_legs_coef', other_legs_coef),
                            ('teacher_stand_threshold', teacher_stand_threshold),
                            ('teacher_forward_threshold', teacher_forward_threshold),
                            ('teacher_other_command_tolerance', teacher_other_command_tolerance)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError('Expected finite nonnegative ' + name)
        if not isinstance(teacher_max_samples, int) or teacher_max_samples < 1:
            raise ValueError('teacher_max_samples must be a positive integer')
        self.rear_teacher_coef = rear_teacher_coef
        self.other_legs_coef = other_legs_coef
        self.teacher_max_samples = teacher_max_samples
        self.teacher_stand_threshold = teacher_stand_threshold
        self.teacher_forward_threshold = teacher_forward_threshold
        self.teacher_other_command_tolerance = teacher_other_command_tolerance
        self.teacher = None
        self.distillation_metrics = {}

    def set_teacher(self, teacher, mirror, dof_names, command_scale):
        """Attach an independently loaded, pinned teacher; never copy the student here."""
        student_ptrs = {p.data_ptr() for p in self.actor_critic.parameters()}
        if any(p.data_ptr() in student_ptrs for p in teacher.parameters()):
            raise ValueError('Teacher must not share student parameters')
        if any(p.device != torch.device(self.device) for p in teacher.parameters()):
            raise ValueError('Teacher must already be on the training device')
        names = list(dof_names)
        rear_names = ['roll_leg_b_r_joint', 'pitch01_leg_b_r_joint', 'pitch02_leg_b_r_joint']
        self.rear_indices = torch.tensor([names.index(n) for n in rear_names], device=self.device)
        self.other_indices = torch.tensor([i for i in range(len(names)) if i not in self.rear_indices.tolist()], device=self.device)
        if len(names) != 12 or len(self.other_indices) != 9:
            raise ValueError('Expected twelve named joints and exactly three RH joints')
        self.command_scale = torch.as_tensor(command_scale, device=self.device).detach().clone()
        if self.command_scale.shape != (3,) or not torch.isfinite(self.command_scale).all() or not (self.command_scale > 0).all():
            raise ValueError('Invalid command normalization')
        self.mirror = mirror
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.teacher = teacher

    def command_mask(self, observations):
        commands = observations[:, :3] / self.command_scale
        standing = (commands.abs() <= self.teacher_stand_threshold).all(dim=1)
        straight = ((commands[:, 0] >= self.teacher_forward_threshold)
                    & (commands[:, 1:3].abs() <= self.teacher_other_command_tolerance).all(dim=1))
        return standing | straight

    @torch.no_grad()
    def teacher_targets(self, observations):
        if self.teacher is None:
            raise RuntimeError('Attach the pinned gait150 teacher before training new v3')
        # Mirror ALL six history frames; query the RAW teacher, not its left-command wrapper.
        mirrored_actions = self.teacher.act_inference(self.mirror.history(observations))
        rear_target = self.mirror.joints(mirrored_actions).index_select(1, self.rear_indices)
        original_actions = self.teacher.act_inference(observations)
        other_target = original_actions.index_select(1, self.other_indices)
        return rear_target, other_target

    def distillation_loss(self, observations, action_mean):
        """Deterministic-action MSE; no reward, action replacement, or teacher gradient."""
        zero = action_mean.sum() * 0.
        if self.rear_teacher_coef == 0 and self.other_legs_coef == 0:
            return zero, dict(rear_mse=0., other_mse=0., weighted_loss=0., samples=0)
        if self.teacher is None:
            raise RuntimeError('New-v3 training requires the pinned frozen teacher')
        ids = self.command_mask(observations).nonzero(as_tuple=False).flatten()
        if len(ids) > self.teacher_max_samples:
            ids = ids[torch.randperm(len(ids), device=ids.device)[:self.teacher_max_samples]]
        if len(ids) == 0:
            return zero, dict(rear_mse=0., other_mse=0., weighted_loss=0., samples=0)
        rear_target, other_target = self.teacher_targets(observations[ids])
        if not torch.isfinite(rear_target).all() or not torch.isfinite(other_target).all():
            raise RuntimeError('Nonfinite frozen teacher target')
        mean = action_mean[ids]
        rear_mse = (mean.index_select(1, self.rear_indices) - rear_target).square().mean()
        other_mse = (mean.index_select(1, self.other_indices) - other_target).square().mean()
        loss = self.rear_teacher_coef * rear_mse + self.other_legs_coef * other_mse
        return loss, dict(rear_mse=rear_mse.item(), other_mse=other_mse.item(),
                          weighted_loss=loss.item(), samples=len(ids))

    def update(self):
        # Preserve HIMPPO's PPO/estimator update equations; add only the actor loss.
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
            estimation_loss, swap_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch, lr=self.learning_rate)
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
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
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
