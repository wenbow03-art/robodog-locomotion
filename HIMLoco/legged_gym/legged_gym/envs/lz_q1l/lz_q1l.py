import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class LZQ1LRobot(LeggedRobot):
    """LeggedRobot with explicit standing commands and command-aware gait rewards."""

    def build_inference_policy(self, actor):
        """One default for playback, recording and standard evaluation."""
        from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror, LeftMirroredLZQ1LPolicy
        device = next(actor.parameters()).device
        mirror = LZQ1LMirror.from_env(self, device=device)
        return LeftMirroredLZQ1LPolicy(actor, mirror, self.commands_scale)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return
        cfg = self.cfg.commands
        u = torch.rand(len(env_ids), device=self.device)
        p_zero = cfg.zero_command_prob
        p_yaw = p_zero + getattr(cfg, "pure_yaw_prob", 0.)
        p_lat = p_yaw + getattr(cfg, "pure_lateral_prob", 0.)
        p_fwd = p_lat + getattr(cfg, "pure_forward_prob", 0.)

        zero = env_ids[u < p_zero]
        self.commands[zero, :3] = 0.

        # In-place turning and pure side-stepping are rare under uniform sampling
        yaw = env_ids[(u >= p_zero) & (u < p_yaw)]
        self.commands[yaw, :2] = 0.
        self.commands[yaw, 2] = self._signed_uniform(len(yaw), 0.3, self.command_ranges["ang_vel_yaw"][1])

        lat = env_ids[(u >= p_yaw) & (u < p_lat)]
        self.commands[lat, 0] = 0.
        # V7 starts with slower dedicated lateral commands. Older tasks keep
        # their original bounds and consume exactly the same random samples.
        lat_low, lat_high = getattr(
            cfg, "pure_lateral_speed_range", (0.2, self.command_ranges["lin_vel_y"][1])
        )
        if not 0.1 < lat_low <= lat_high:
            raise ValueError("pure_lateral_speed_range must satisfy 0.1 < low <= high")
        self.commands[lat, 1] = self._signed_uniform(len(lat), lat_low, lat_high)
        self.commands[lat, 2] = 0.

        # Allocate positive, straight-ahead commands from the remaining random pool.
        # Keep older versions' sampling and random-number consumption unchanged.
        if p_fwd > p_lat:
            fwd = env_ids[(u >= p_lat) & (u < p_fwd)]
            self.commands[fwd, 0] = 0.2 + (self.command_ranges["lin_vel_x"][1] - 0.2) * torch.rand(len(fwd), device=self.device)
            self.commands[fwd, 1:3] = 0.

    def _signed_uniform(self, n, low, high):
        mag = low + (high - low) * torch.rand(n, device=self.device)
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1., 1.)
        return mag * sign

    def _is_moving_command(self):
        return (torch.norm(self.commands[:, :2], dim=1) > 0.1) | (torch.abs(self.commands[:, 2]) > 0.1)

    def _reward_tracking_lateral(self):
        # Additional tracking incentive for pure side-stepping; the existing
        # planar and yaw rewards still penalize forward drift and unwanted yaw.
        lateral = (
            (self.commands[:, 1].abs() > 0.1)
            & (self.commands[:, 0].abs() < 0.05)
            & (self.commands[:, 2].abs() < 0.05)
        )
        error = self.commands[:, 1] - self.base_lin_vel[:, 1]
        return torch.exp(-error.square() / self.cfg.rewards.lateral_tracking_sigma) * lateral

    def _reward_lateral_error(self):
        # Positive normalized error; a negative config scale makes it a penalty.
        # Enabled only for pure lateral commands, matching tracking_lateral.
        cmd = self.commands[:, 1]
        lateral = (
            (cmd.abs() > 0.1)
            & (self.commands[:, 0].abs() < 0.05)
            & (self.commands[:, 2].abs() < 0.05)
        )
        error = (cmd - self.base_lin_vel[:, 1]).abs()
        return lateral.float() * error / cmd.abs().clamp_min(0.15)

    def _reward_stand_still(self):
        # Penalize leaving the default pose only when every command component is zero
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * ~self._is_moving_command()

    def _reward_feet_air_time(self):
        # PhysX contact reporting is noisy on meshes, so contacts are filtered over two steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_air_time = torch.sum((self.feet_air_time - self.cfg.rewards.feet_air_time_target) * first_contact, dim=1)
        rew_air_time *= self._is_moving_command()
        self.feet_air_time *= ~contact_filt
        return rew_air_time
