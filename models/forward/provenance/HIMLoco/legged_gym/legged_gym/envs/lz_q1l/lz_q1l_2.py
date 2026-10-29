"""Environment logic for the new LZ-Q1L lineage, configured by lz_q1l_2_config.

New V1 provides command sampling and a standing-only unsupported-foot penalty.
Extend this module for subsequent new-lineage environments.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from isaacgym.torch_utils import quat_apply
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from .lz_q1l import LZQ1LRobot


class LZQ1LV1SupportRobot(LeggedRobot):
    # Preserve the project's normal inference factory without inheriting V2 rewards.
    build_inference_policy = LZQ1LRobot.build_inference_policy

    def _init_buffers(self):
        super()._init_buffers()
        if self.cfg.terrain.mesh_type != 'plane' or self.cfg.commands.heading_command:
            raise ValueError('V1 support task requires a flat plane and yaw-rate commands')
        cmd = self.cfg.commands
        if not (0 <= cmd.zero_command_prob <= 1 and 0 <= cmd.pure_forward_prob
                and cmd.zero_command_prob + cmd.pure_forward_prob <= 1):
            raise ValueError('Invalid standing/forward command probabilities')
        lo, hi = cmd.pure_forward_speed_range
        if not (0 < lo <= hi):
            raise ValueError('Pure forward speed range must be positive and ordered')
        cfg = self.cfg.rewards
        if min(cfg.stand_support_grace_s, cfg.stand_support_air_time_s,
               cfg.stand_support_height_m, cfg.stand_support_contact_force_n,
               cfg.stand_command_threshold) < 0:
            raise ValueError('Support thresholds must be nonnegative')
        self.support_grace_steps = math.ceil(cfg.stand_support_grace_s / self.dt)
        self.support_air_steps = math.ceil(cfg.stand_support_air_time_s / self.dt)
        shape = (self.num_envs, len(self.feet_indices))
        self.support_stand_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.support_offground_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.support_last_contacts = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.support_penalized_feet = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.support_foot_gap = torch.zeros(shape, device=self.device)
        self.command_sample_mode = torch.full((self.num_envs,), 2, dtype=torch.long, device=self.device)

        # Actual collision-sphere bottom, not the foot-link origin or visual mesh.
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        path = Path(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        xml = ET.parse(path).getroot()
        offsets, radii = [], []
        for index in self.feet_indices.tolist():
            collisions = xml.findall("link[@name='%s']/collision" % names[index])
            if len(collisions) != 1 or collisions[0].find('geometry/sphere') is None:
                raise ValueError('Support height calculation requires one sphere per foot')
            collision = collisions[0]
            origin = collision.find('origin')
            offsets.append([float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()])
            radii.append(float(collision.find('geometry/sphere').get('radius')))
        self.support_sphere_offsets = torch.tensor(offsets, device=self.device).expand(self.num_envs, -1, -1)
        self.support_sphere_radii = torch.tensor(radii, device=self.device)

    def _clear_support(self, ids):
        self.support_stand_steps[ids] = 0
        self.support_offground_steps[ids] = 0
        self.support_last_contacts[ids] = False
        self.support_penalized_feet[ids] = False

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        # Mixed commands respect configured/curriculum ranges (no hardcoded vx).
        for axis, name in enumerate(('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw')):
            lo, hi = self.command_ranges[name]
            self.commands[env_ids, axis] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        self.commands[env_ids, 3] = 0.
        high = env_ids[env_ids < self.num_envs * .2]
        self.commands[high, 1] *= self.commands[high, 0].abs() < 1.
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > .2).unsqueeze(1)

        cfg = self.cfg.commands
        u = torch.rand(len(env_ids), device=self.device)
        stand = env_ids[u < cfg.zero_command_prob]
        forward = env_ids[(u >= cfg.zero_command_prob) & (u < cfg.zero_command_prob + cfg.pure_forward_prob)]
        # Override after the legacy small-command cutoff; slow forward stays forward.
        self.commands[stand] = 0.
        lo = max(cfg.pure_forward_speed_range[0], self.command_ranges['lin_vel_x'][0])
        hi = min(cfg.pure_forward_speed_range[1], self.command_ranges['lin_vel_x'][1])
        if cfg.pure_forward_prob > 0 and lo > hi:
            raise ValueError('Pure forward range does not overlap configured vx range')
        self.commands[forward] = 0.
        self.commands[forward, 0] = lo + (hi - lo) * torch.rand(len(forward), device=self.device)
        self.command_sample_mode[env_ids] = 2
        self.command_sample_mode[stand] = 0
        self.command_sample_mode[forward] = 1
        self._clear_support(env_ids)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_standing_support()

    def _update_standing_support(self):
        cfg = self.cfg.rewards
        standing = (self.commands[:, :3].abs() <= cfg.stand_command_threshold).all(dim=1)
        self.support_stand_steps = torch.where(standing, self.support_stand_steps + 1, 0)
        body = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        offset = quat_apply(body[:, :, 3:7].reshape(-1, 4), self.support_sphere_offsets.reshape(-1, 3))
        self.support_foot_gap = body[:, :, 2] + offset.reshape(self.num_envs, -1, 3)[:, :, 2] - self.support_sphere_radii
        contact = self.contact_forces[:, self.feet_indices, 2] > cfg.stand_support_contact_force_n
        contact_filtered = contact | self.support_last_contacts
        self.support_last_contacts.copy_(contact)
        eligible = standing & (self.support_stand_steps > self.support_grace_steps)
        offground = ((self.support_foot_gap > cfg.stand_support_height_m)
                     & ~contact_filtered & eligible[:, None])
        self.support_offground_steps = torch.where(offground, self.support_offground_steps + 1, 0)
        self.support_penalized_feet = self.support_offground_steps > self.support_air_steps

    def _reward_stand_unsupported(self):
        # State updates happen once per physics callback, never as a reward side effect.
        return self.support_penalized_feet.float().sum(dim=1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._clear_support(env_ids)


class LZQ1LV2UnifiedRobot(LZQ1LV1SupportRobot):
    """New V2: dedicated movement commands plus passive gait diagnostics."""

    command_names = ('stand', 'forward', 'left', 'right', 'yaw_left', 'yaw_right')
    _reward_tracking_lateral = LZQ1LRobot._reward_tracking_lateral

    def _init_buffers(self):
        cfg = self.cfg.commands
        probabilities = (cfg.zero_command_prob, cfg.pure_forward_prob,
                         cfg.pure_lateral_prob, cfg.pure_yaw_prob)
        if any(not math.isfinite(p) or p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1., abs_tol=1e-8):
            raise ValueError('V2 command probabilities must be nonnegative and sum to one')
        if cfg.curriculum:
            raise ValueError('Initial V2 uses fixed command ranges, without curriculum')
        for key, bounds, symmetric in (
            ('pure_forward_speed_range', cfg.ranges.lin_vel_x, False),
            ('pure_lateral_speed_range', cfg.ranges.lin_vel_y, True),
            ('pure_yaw_speed_range', cfg.ranges.ang_vel_yaw, True),
        ):
            lo, hi = getattr(cfg, key)
            if not (math.isfinite(lo) and math.isfinite(hi) and .1 < lo <= hi
                    and hi <= bounds[1] and (-hi if symmetric else lo) >= bounds[0]):
                raise ValueError('Invalid V2 command range: ' + key)
        super()._init_buffers()
        self.v2_air_steps = torch.zeros_like(self.support_offground_steps)
        self.v2_metric_count = torch.zeros(6, device=self.device)
        self.v2_metric_sums = {
            key: torch.zeros(6, width, device=self.device) for key, width in (
                ('velocity', 3), ('abs_velocity_error', 3), ('force_z_N', 4),
                ('contact', 4), ('gap_m', 4), ('contact_slip_m_s', 4),
                ('fall', 1), ('timeout', 1))}
        self.v2_air_max_s = torch.zeros(6, 4, device=self.device)
        self.v2_gap_max_m = torch.zeros(6, 4, device=self.device)

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        cfg = self.cfg.commands
        u = torch.rand(len(env_ids), device=self.device)
        forward_start = cfg.zero_command_prob
        lateral_start = forward_start + cfg.pure_forward_prob
        yaw_start = lateral_start + cfg.pure_lateral_prob
        self.commands[env_ids] = 0.
        self.command_sample_mode[env_ids] = 0
        for selected, axis, bounds, signed, positive_mode, negative_mode in (
            ((u >= forward_start) & (u < lateral_start), 0, cfg.pure_forward_speed_range, False, 1, 1),
            ((u >= lateral_start) & (u < yaw_start), 1, cfg.pure_lateral_speed_range, True, 2, 3),
            (u >= yaw_start, 2, cfg.pure_yaw_speed_range, True, 4, 5),
        ):
            ids = env_ids[selected]
            lo, hi = bounds
            value = lo + (hi - lo) * torch.rand(len(ids), device=self.device)
            if signed:
                value *= torch.where(torch.rand(len(ids), device=self.device) < .5, -1., 1.)
            self.commands[ids, axis] = value
            self.command_sample_mode[ids] = torch.where(value > 0, positive_mode, negative_mode)
        # Write dedicated commands directly: the base sampler's 0.2 m/s
        # deadband must not erase the new 0.15--0.25 m/s lateral commands.
        self._clear_support(env_ids)
        self.v2_air_steps[env_ids] = 0

    def compute_reward(self):
        super().compute_reward()
        # Capture before reset_idx clears commands/support state. Diagnostics
        # do not enter observations, rewards, or the optimizer.
        modes = self.command_sample_mode
        force = self.contact_forces[:, self.feet_indices, 2]
        contact = force > self.cfg.rewards.stand_support_contact_force_n
        self.v2_air_steps = torch.where(contact, 0, self.v2_air_steps + 1)
        body = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        offset = quat_apply(body[:, :, 3:7].reshape(-1, 4), self.support_sphere_offsets.reshape(-1, 3)).reshape(self.num_envs, 4, 3)
        offset[:, :, 2] -= self.support_sphere_radii
        # Estimated sphere-bottom velocity in the world frame, relative to
        # the fixed plane. Body angular velocity matters when a foot rotates.
        point_velocity = body[:, :, 7:10] + torch.cross(body[:, :, 10:13], offset, dim=-1)
        slip = torch.norm(point_velocity[:, :, :2], dim=-1) * contact
        velocity = torch.cat((self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]), dim=1)
        values = dict(velocity=velocity, abs_velocity_error=(velocity - self.commands[:, :3]).abs(),
                      force_z_N=force, contact=contact.float(), gap_m=self.support_foot_gap,
                      contact_slip_m_s=slip,
                      fall=(self.reset_buf.bool() & ~self.time_out_buf).float()[:, None],
                      timeout=self.time_out_buf.float()[:, None])
        self.v2_metric_count.index_add_(0, modes, torch.ones(self.num_envs, device=self.device))
        for key, value in values.items():
            self.v2_metric_sums[key].index_add_(0, modes, value)
        air_seconds = self.v2_air_steps.to(self.support_foot_gap.dtype) * float(self.dt)
        for mode in range(6):
            mask = modes[:, None] == mode
            self.v2_air_max_s[mode] = torch.maximum(self.v2_air_max_s[mode],
                air_seconds.masked_fill(~mask, 0.).max(dim=0).values)
            self.v2_gap_max_m[mode] = torch.maximum(self.v2_gap_max_m[mode],
                self.support_foot_gap.masked_fill(~mask, 0.).max(dim=0).values)

    def take_diagnostics(self):
        """Drain raw-policy training statistics, grouped by sampled direction.

        Forces are 50 Hz snapshots, not integrated physics-step impulses.
        Falls are event counts per exposure, not independent-trial success rates.
        Maximum noncontact duration is scoped to a continuous command segment.
        """
        counts = self.v2_metric_count.cpu().tolist()
        sums = {key: value.cpu().tolist() for key, value in self.v2_metric_sums.items()}
        air, gap = self.v2_air_max_s.cpu().tolist(), self.v2_gap_max_m.cpu().tolist()
        result = {}
        for i, name in enumerate(self.command_names):
            n = counts[i]
            if not n:
                continue
            row = {key: [v / n for v in sums[key][i]] for key in
                   ('velocity', 'abs_velocity_error', 'force_z_N', 'contact', 'gap_m')}
            row.update(samples=int(n), exposure_env_seconds=n * self.dt,
                       fall_events=int(sums['fall'][i][0]), timeout_events=int(sums['timeout'][i][0]),
                       longest_noncontact_s=air[i], max_gap_m=gap[i],
                       contact_slip_m_s=[sums['contact_slip_m_s'][i][j] / sums['contact'][i][j]
                           if sums['contact'][i][j] else None for j in range(4)])
            result[name] = row
        self.v2_metric_count.zero_()
        for value in self.v2_metric_sums.values():
            value.zero_()
        self.v2_air_max_s.zero_()
        self.v2_gap_max_m.zero_()
        return result


class LZQ1LV2GaitRobot(LZQ1LV2UnifiedRobot):
    """Three straight-walking penalties; no change to observations or PD control."""

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.rewards
        for name in ('gait_min_forward_command', 'swing_clearance_target_m', 'swing_unloaded_force_n',
                     'swing_speed_scale_m_s', 'slide_contact_force_n', 'slide_speed_scale_m_s',
                     'slide_normalized_speed_cap', 'rear_force_window_s',
                     'rear_force_denominator_epsilon_n', 'rear_force_min_total_n'):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Expected positive finite gait setting: ' + name)
        if not (0 <= cfg.rear_force_imbalance_tolerance < 1):
            raise ValueError('Rear-force tolerance must be in [0, 1)')
        if min(cfg.gait_other_command_tolerance, cfg.gait_transition_grace_s,
               cfg.slide_speed_deadband_m_s) < 0:
            raise ValueError('Gait thresholds must be nonnegative')
        self.rear_force_window_steps = max(1, math.ceil(cfg.rear_force_window_s / self.dt))
        self.gait_grace_steps = math.ceil(cfg.gait_transition_grace_s / self.dt)
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        feet = [names[i] for i in self.feet_indices.tolist()]
        self.rear_foot_slots = torch.tensor([feet.index('pitch03_leg_b_l_link'),
                                            feet.index('pitch03_leg_b_r_link')], device=self.device)
        # About 1.6 MiB for 4096 environments and a 50-sample window.
        self.rear_force_history = torch.zeros(self.rear_force_window_steps, self.num_envs, 2, device=self.device)
        self.rear_force_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self.rear_force_mean = torch.zeros_like(self.rear_force_sum)
        self.rear_force_cursor = 0
        self.gait_forward_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_last_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.gait_penalties = torch.zeros(self.num_envs, 3, device=self.device)

    def _clear_gait_state(self, ids):
        if len(ids) == 0 or not hasattr(self, 'rear_force_history'):
            return
        self.rear_force_history[:, ids] = 0.
        self.rear_force_sum[ids] = 0.
        self.rear_force_mean[ids] = 0.
        self.gait_forward_steps[ids] = 0
        self.gait_penalties[ids] = 0.
        self.gait_last_commands[ids] = self.commands[ids, :3]

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self._clear_gait_state(env_ids)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._clear_gait_state(env_ids)

    @staticmethod
    def gait_penalty_values(force_z, gap_m, relative_foot_speed, ground_foot_speed, rear_force_mean, cfg):
        """Nonnegative per-environment terms; scales supply the negative signs.

        The swing term applies to low-load moving feet, including light-touch
        scuffing. It penalizes clearance deficits, not heights above the target.
        Inputs have named-foot ordering only for rear_force_mean (left, right).
        """
        deficit = ((cfg.swing_clearance_target_m - gap_m) / cfg.swing_clearance_target_m).clamp(0., 1.)
        swing_weight = (force_z <= cfg.swing_unloaded_force_n).float()
        swing_weight *= (relative_foot_speed / cfg.swing_speed_scale_m_s).clamp(0., 1.)
        clearance = (deficit.square() * swing_weight).mean(dim=-1)
        slip = ((ground_foot_speed - cfg.slide_speed_deadband_m_s) / cfg.slide_speed_scale_m_s)
        slip = slip.clamp(0., cfg.slide_normalized_speed_cap).square()
        slide = (slip * (force_z > cfg.slide_contact_force_n)).mean(dim=-1)
        rear = rear_force_mean.clamp_min(0.)
        total = rear.sum(dim=-1)
        imbalance = (rear[:, 0] - rear[:, 1]).abs() / (total + cfg.rear_force_denominator_epsilon_n)
        balance = (imbalance - cfg.rear_force_imbalance_tolerance).clamp_min(0.).square()
        # Without this guard, unloading BOTH rear feet would have zero variance.
        insufficient = ((cfg.rear_force_min_total_n - total) / cfg.rear_force_min_total_n).clamp(0., 1.)
        balance += insufficient.square()
        return torch.stack((clearance, slide, balance), dim=-1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_gait_rewards()

    def _update_gait_rewards(self):
        """Update history exactly once, before reward evaluation and resets."""
        cfg = self.cfg.rewards
        changed = (self.commands[:, :3] - self.gait_last_commands).abs().amax(dim=1) > 1e-6
        self._clear_gait_state(changed.nonzero(as_tuple=False).flatten())
        self.gait_last_commands.copy_(self.commands[:, :3])
        straight = ((self.commands[:, 0] >= cfg.gait_min_forward_command)
                    & (self.commands[:, 1:3].abs() <= cfg.gait_other_command_tolerance).all(dim=1))
        self.gait_forward_steps = torch.where(straight, self.gait_forward_steps + 1, 0)
        force = self.contact_forces[:, self.feet_indices, 2].clamp_min(0.)
        new = force[:, self.rear_foot_slots] * straight[:, None]
        old = self.rear_force_history[self.rear_force_cursor]
        self.rear_force_sum.add_(new - old)
        old.copy_(new)
        self.rear_force_cursor = (self.rear_force_cursor + 1) % self.rear_force_window_steps
        self.rear_force_mean.copy_(self.rear_force_sum / self.rear_force_window_steps)

        body = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices]
        center_offset = quat_apply(body[:, :, 3:7].reshape(-1, 4),
            self.support_sphere_offsets.reshape(-1, 3)).reshape(self.num_envs, 4, 3)
        center_position = body[:, :, :3] + center_offset
        center_velocity = body[:, :, 7:10] + torch.cross(body[:, :, 10:13], center_offset, dim=-1)
        # Relative to the translating AND rotating base, expressed in world axes.
        relative_velocity = center_velocity - self.root_states[:, None, 7:10]
        relative_velocity -= torch.cross(self.root_states[:, None, 10:13].expand(-1, 4, -1),
            center_position - self.root_states[:, None, :3], dim=-1)
        bottom_offset = center_offset.clone()
        bottom_offset[:, :, 2] -= self.support_sphere_radii
        bottom_velocity = body[:, :, 7:10] + torch.cross(body[:, :, 10:13], bottom_offset, dim=-1)
        values = self.gait_penalty_values(force, self.support_foot_gap,
            relative_velocity[:, :, :2].norm(dim=-1), bottom_velocity[:, :, :2].norm(dim=-1),
            self.rear_force_mean, cfg)
        moving_ready = straight & (self.gait_forward_steps > self.gait_grace_steps)
        values[:, :2] *= moving_ready[:, None]
        values[:, 2] *= straight & (self.gait_forward_steps >= self.rear_force_window_steps)
        self.gait_penalties.copy_(values)

    def _reward_swing_clearance(self):
        return self.gait_penalties[:, 0]

    def _reward_feet_slide(self):
        return self.gait_penalties[:, 1]

    def _reward_rear_force_balance(self):
        return self.gait_penalties[:, 2]


class LZQ1LV2GaitHeightRobot(LZQ1LV2GaitRobot):
    """Standing load distribution; walking rewards retain their parent behavior."""

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.rewards
        for name in ('stand_force_window_s', 'stand_pair_force_epsilon_n'):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError('Expected positive finite setting: ' + name)
        if not math.isfinite(cfg.stand_force_grace_s) or cfg.stand_force_grace_s < 0:
            raise ValueError('Standing force grace must be finite and nonnegative')
        if not 0 <= cfg.stand_pair_force_tolerance < 1:
            raise ValueError('Standing force tolerance must be in [0, 1)')
        if not 0 < cfg.stand_min_load_fraction <= .25:
            raise ValueError('Minimum foot load fraction must be in (0, 0.25]')
        self.stand_force_window_steps = max(1, math.ceil(cfg.stand_force_window_s / self.dt))
        self.stand_force_ready_steps = max(self.stand_force_window_steps,
                                          math.ceil(cfg.stand_force_grace_s / self.dt))
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        feet = [names[i] for i in self.feet_indices.tolist()]
        self.stand_force_slots = torch.tensor([feet.index(name) for name in (
            'pitch03_leg_b_l_link', 'pitch03_leg_b_r_link',
            'pitch03_leg_f_l_link', 'pitch03_leg_f_r_link')], device=self.device)
        self.stand_force_history = torch.zeros(self.stand_force_window_steps, self.num_envs, 4, device=self.device)
        self.stand_force_sum = torch.zeros(self.num_envs, 4, device=self.device)
        self.stand_force_mean = torch.zeros_like(self.stand_force_sum)
        self.stand_force_cursor = 0
        self.stand_force_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stand_force_last_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.stand_force_penalties = torch.zeros(self.num_envs, 2, device=self.device)

    def _clear_stand_force(self, ids):
        if len(ids) == 0 or not hasattr(self, 'stand_force_history'):
            return
        self.stand_force_history[:, ids] = 0.
        self.stand_force_sum[ids] = 0.
        self.stand_force_mean[ids] = 0.
        self.stand_force_steps[ids] = 0
        self.stand_force_penalties[ids] = 0.
        self.stand_force_last_commands[ids] = self.commands[ids, :3]

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self._clear_stand_force(env_ids)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._clear_stand_force(env_ids)

    @staticmethod
    def stand_force_penalty_values(mean_force, cfg):
        """Positive costs, inputs ordered LH/RH/LF/RF; scales and dt applied by base."""
        force = mean_force.clamp_min(0.)
        pairs = force.reshape(-1, 2, 2)
        imbalance = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
        imbalance /= pairs.sum(dim=-1) + cfg.stand_pair_force_epsilon_n
        balance = (imbalance - cfg.stand_pair_force_tolerance).clamp_min(0.).square().mean(dim=-1)
        # All-zero support yields shares=0 and maximum low-load cost, not 0/0.
        shares = force / force.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        deficit = ((cfg.stand_min_load_fraction - shares) / cfg.stand_min_load_fraction).clamp(0., 1.)
        return torch.stack((balance, deficit.square().mean(dim=-1)), dim=-1)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_stand_force_rewards()

    def _update_stand_force_rewards(self):
        cfg = self.cfg.rewards
        changed = (self.commands[:, :3] - self.stand_force_last_commands).abs().amax(dim=1) > 1e-6
        self._clear_stand_force(changed.nonzero(as_tuple=False).flatten())
        self.stand_force_last_commands.copy_(self.commands[:, :3])
        standing = (self.commands[:, :3].abs() <= cfg.stand_command_threshold).all(dim=1)
        self.stand_force_steps = torch.where(standing, self.stand_force_steps + 1, 0)
        force = self.contact_forces[:, self.feet_indices, 2][:, self.stand_force_slots].clamp_min(0.)
        new = force * standing[:, None]
        old = self.stand_force_history[self.stand_force_cursor]
        self.stand_force_sum.add_(new - old)
        old.copy_(new)
        self.stand_force_cursor = (self.stand_force_cursor + 1) % self.stand_force_window_steps
        self.stand_force_mean.copy_(self.stand_force_sum / self.stand_force_window_steps)
        values = self.stand_force_penalty_values(self.stand_force_mean, cfg)
        values *= (standing & (self.stand_force_steps >= self.stand_force_ready_steps))[:, None]
        self.stand_force_penalties.copy_(values)

    def _reward_stand_pair_force_balance(self):
        return self.stand_force_penalties[:, 0]

    def _reward_stand_low_load(self):
        return self.stand_force_penalties[:, 1]


class LZQ1LV2StandFrontRobot(LZQ1LV2GaitHeightRobot):
    """Preserve gait rewards; add command-specific posture and front-pair support."""

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.rewards
        for name in ('front_force_window_s', 'front_force_denominator_epsilon_n', 'front_force_min_total_n'):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError('Expected positive finite setting: ' + name)
        if not math.isfinite(cfg.stand_posture_grace_s) or cfg.stand_posture_grace_s < 0:
            raise ValueError('Standing posture grace must be finite and nonnegative')
        if not 0 <= cfg.front_force_imbalance_tolerance < 1:
            raise ValueError('Front-force tolerance must be in [0, 1)')
        self.stand_posture_grace_steps = math.ceil(cfg.stand_posture_grace_s / self.dt)
        self.front_force_window_steps = max(1, math.ceil(cfg.front_force_window_s / self.dt))
        self.front_foot_slots = self.stand_force_slots[2:]
        self.front_force_history = torch.zeros(self.front_force_window_steps, self.num_envs, 2, device=self.device)
        self.front_force_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self.front_force_mean = torch.zeros_like(self.front_force_sum)
        self.front_force_cursor = 0
        self.front_force_penalty = torch.zeros(self.num_envs, device=self.device)

    def _clear_gait_state(self, ids):
        # Parent invokes this on reset, resampling and external command changes.
        super()._clear_gait_state(ids)
        if len(ids) and hasattr(self, 'front_force_history'):
            self.front_force_history[:, ids] = 0.
            self.front_force_sum[ids] = 0.
            self.front_force_mean[ids] = 0.
            self.front_force_penalty[ids] = 0.

    @staticmethod
    def front_force_penalty_value(mean_force, cfg):
        force = mean_force.clamp_min(0.)
        total = force.sum(dim=-1)
        imbalance = (force[:, 0] - force[:, 1]).abs() / (total + cfg.front_force_denominator_epsilon_n)
        excess = (imbalance - cfg.front_force_imbalance_tolerance).clamp_min(0.)
        insufficient = ((cfg.front_force_min_total_n - total) / cfg.front_force_min_total_n).clamp(0., 1.)
        return excess.square() + insufficient.square()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_front_force_rewards()

    def _update_front_force_rewards(self):
        cfg = self.cfg.rewards
        straight = ((self.commands[:, 0] >= cfg.gait_min_forward_command)
                    & (self.commands[:, 1:3].abs() <= cfg.gait_other_command_tolerance).all(dim=1))
        force = self.contact_forces[:, self.feet_indices, 2][:, self.front_foot_slots].clamp_min(0.)
        new = force * straight[:, None]
        old = self.front_force_history[self.front_force_cursor]
        self.front_force_sum.add_(new - old)
        old.copy_(new)
        self.front_force_cursor = (self.front_force_cursor + 1) % self.front_force_window_steps
        self.front_force_mean.copy_(self.front_force_sum / self.front_force_window_steps)
        values = self.front_force_penalty_value(self.front_force_mean, cfg)
        values *= straight & (self.gait_forward_steps >= self.front_force_window_steps)
        self.front_force_penalty.copy_(values)

    def _stand_posture_ready(self):
        return ((self.commands[:, :3].abs() <= self.cfg.rewards.stand_command_threshold).all(dim=1)
                & (self.stand_force_steps >= self.stand_posture_grace_steps))

    def _reward_stand_height_extra(self):
        return self._reward_base_height() * self._stand_posture_ready()

    def _reward_stand_orientation_extra(self):
        return self._reward_orientation() * self._stand_posture_ready()

    def _reward_front_force_balance(self):
        return self.front_force_penalty


class LZQ1LV2BalanceRollRobot(LZQ1LV2GaitHeightRobot):
    """Front/rear support balance and roll-only posture cost; source height unchanged."""

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.rewards
        for name in ('front_force_window_s', 'front_force_denominator_epsilon_n', 'front_force_min_total_n'):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError('Expected positive finite setting: ' + name)
        if not 0 <= cfg.front_force_imbalance_tolerance < 1:
            raise ValueError('Front-force tolerance must be in [0, 1)')
        self.front_force_window_steps = max(1, math.ceil(cfg.front_force_window_s / self.dt))
        self.front_foot_slots = self.stand_force_slots[2:]
        self.front_force_history = torch.zeros(self.front_force_window_steps, self.num_envs, 2, device=self.device)
        self.front_force_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self.front_force_mean = torch.zeros_like(self.front_force_sum)
        self.front_force_cursor = 0
        self.front_force_penalty = torch.zeros(self.num_envs, device=self.device)

    def _clear_gait_state(self, ids):
        super()._clear_gait_state(ids)
        if len(ids) and hasattr(self, 'front_force_history'):
            self.front_force_history[:, ids] = 0.
            self.front_force_sum[ids] = 0.
            self.front_force_mean[ids] = 0.
            self.front_force_penalty[ids] = 0.

    # Reuse the tested front-pair formula and window without standing extras.
    front_force_penalty_value = staticmethod(LZQ1LV2StandFrontRobot.front_force_penalty_value)
    _update_front_force_rewards = LZQ1LV2StandFrontRobot._update_front_force_rewards
    _reward_front_force_balance = LZQ1LV2StandFrontRobot._reward_front_force_balance

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_front_force_rewards()

    def _reward_body_roll(self):
        # atan2 removes pitch-dependent scaling of gravity_y; no extra pitch cost.
        roll = torch.atan2(-self.projected_gravity[:, 1], -self.projected_gravity[:, 2])
        return roll.square()


class LZQ1LV2SideLoadRobot(LZQ1LV2GaitRobot):
    """Penalize only excess LEFT support during stand and forward."""

    def _init_buffers(self):
        super()._init_buffers()
        cfg = self.cfg.rewards
        for name in ('side_force_window_s', 'side_force_epsilon_n'):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError('Expected positive finite setting: ' + name)
        names = self.gym.get_actor_rigid_body_names(self.envs[0], self.actor_handles[0])
        feet = [names[i] for i in self.feet_indices.tolist()]
        self.side_foot_slots = torch.tensor([feet.index('pitch03_leg_b_l_link'),
            feet.index('pitch03_leg_f_l_link'), feet.index('pitch03_leg_b_r_link'),
            feet.index('pitch03_leg_f_r_link')], device=self.device)
        self.side_force_window_steps = max(1, math.ceil(cfg.side_force_window_s / self.dt))
        self.side_force_history = torch.zeros(self.side_force_window_steps, self.num_envs, 2, device=self.device)
        self.side_force_sum = torch.zeros(self.num_envs, 2, device=self.device)
        self.side_force_mean = torch.zeros_like(self.side_force_sum)
        self.side_force_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.side_force_last_commands = self.commands[:, :3].clone()
        self.side_force_penalty = torch.zeros(self.num_envs, device=self.device)
        self.side_force_cursor = 0

    def _clear_side_force(self, ids):
        if len(ids) == 0 or not hasattr(self, 'side_force_history'):
            return
        self.side_force_history[:, ids] = 0.
        self.side_force_sum[ids] = 0.
        self.side_force_mean[ids] = 0.
        self.side_force_steps[ids] = 0
        self.side_force_penalty[ids] = 0.
        self.side_force_last_commands[ids] = self.commands[ids, :3]

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        self._clear_side_force(env_ids)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._clear_side_force(env_ids)

    @staticmethod
    def side_force_penalty_value(mean_sides, cfg):
        sides = mean_sides.clamp_min(0.)
        total = sides.sum(dim=-1)
        excess = (sides[:, 0] - sides[:, 1]).clamp_min(0.) / (total + cfg.side_force_epsilon_n)
        return excess.square()

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_side_force_rewards()

    def _update_side_force_rewards(self):
        cfg = self.cfg.rewards
        changed = (self.commands[:, :3] - self.side_force_last_commands).abs().amax(dim=1) > 1e-6
        self._clear_side_force(changed.nonzero(as_tuple=False).flatten())
        self.side_force_last_commands.copy_(self.commands[:, :3])
        standing = (self.commands[:, :3].abs() <= cfg.stand_command_threshold).all(dim=1)
        straight = ((self.commands[:, 0] >= cfg.gait_min_forward_command)
                    & (self.commands[:, 1:3].abs() <= cfg.gait_other_command_tolerance).all(dim=1))
        eligible = standing | straight
        self.side_force_steps = torch.where(eligible, self.side_force_steps + 1, 0)
        force = self.contact_forces[:, self.feet_indices, 2][:, self.side_foot_slots].clamp_min(0.)
        new = force.reshape(self.num_envs, 2, 2).sum(dim=-1) * eligible[:, None]
        old = self.side_force_history[self.side_force_cursor]
        self.side_force_sum.add_(new - old)
        old.copy_(new)
        self.side_force_cursor = (self.side_force_cursor + 1) % self.side_force_window_steps
        self.side_force_mean.copy_(self.side_force_sum / self.side_force_window_steps)
        values = self.side_force_penalty_value(self.side_force_mean, cfg)
        values *= eligible & (self.side_force_steps >= self.side_force_window_steps)
        self.side_force_penalty.copy_(values)

    def _reward_left_load_excess(self):
        return self.side_force_penalty
