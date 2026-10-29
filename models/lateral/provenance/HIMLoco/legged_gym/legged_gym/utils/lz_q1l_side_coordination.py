"""Per-foot lateral support and landing state, independent of simulator bindings."""
import math
import torch


class SideCoordinationTracker:
    def __init__(self, num_envs, device, dt, cfg):
        self.dt = dt
        self.cfg = cfg
        self.window = int(round(cfg.side_load_window_s / dt))
        self.contact_steps = int(math.ceil(cfg.side_contact_duration_s / dt - 1e-9))
        if self.window < 1 or self.contact_steps < 1:
            raise ValueError('Support windows must contain at least one control step')
        self.history = torch.zeros(self.window, num_envs, 4, device=device)
        self.force_sum = torch.zeros(num_envs, 4, device=device)
        self.samples = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.contact_streak = torch.zeros(num_envs, 4, dtype=torch.long, device=device)
        self.unloaded_time = torch.zeros_like(self.force_sum)
        self.peak_gap = torch.zeros_like(self.force_sum)
        self.armed = torch.zeros_like(self.force_sum, dtype=torch.bool)
        self.had_air = torch.zeros_like(self.armed)
        self.load_share = torch.zeros_like(self.force_sum)
        self.stable_contact = torch.zeros_like(self.armed)
        self.landing_duration = torch.zeros_like(self.force_sum)
        self.head = 0

    def reset(self, ids):
        if len(ids) == 0:
            return
        self.history[:, ids] = 0.
        for name in ('force_sum', 'samples', 'contact_streak', 'unloaded_time',
                     'peak_gap', 'armed', 'had_air', 'load_share',
                     'stable_contact', 'landing_duration'):
            getattr(self, name)[ids] = 0

    def update(self, force_z, collision_gap, active):
        self.reset((~active).nonzero(as_tuple=False).flatten())
        force = force_z.clamp_min(0.) * active[:, None]
        self.force_sum += force - self.history[self.head]
        self.history[self.head].copy_(force)
        self.head = (self.head + 1) % self.window
        self.samples.copy_((self.samples + active.long()).clamp_max(self.window))
        mean_force = self.force_sum.clamp_min(0.) / self.samples.clamp_min(1)[:, None]
        total = mean_force.sum(dim=1)
        self.load_share.copy_(mean_force / total.clamp_min(1e-6)[:, None])
        ready = active & (self.samples >= self.window) & (total >= self.cfg.side_min_total_force_n)
        deficit = ((self.cfg.side_min_load_share - self.load_share) /
                   self.cfg.side_min_load_share).clamp_min(0.)
        load_cost = deficit.square().sum(dim=1) * ready

        contact = (force >= self.cfg.side_contact_force_n) & active[:, None]
        self.contact_streak.copy_(torch.where(contact, self.contact_streak + 1,
                                              torch.zeros_like(self.contact_streak)))
        self.stable_contact.copy_(self.contact_streak >= self.contact_steps)
        self.unloaded_time += self.dt * active[:, None]
        self.had_air |= (~contact) & active[:, None]
        self.peak_gap.copy_(torch.maximum(self.peak_gap, collision_gap.clamp_min(0.)) * active[:, None])
        landing = self.stable_contact & self.had_air & self.armed
        # Remove the contact-confirmation interval from the measured swing.
        duration = (self.unloaded_time - self.contact_streak * self.dt).clamp_min(0.)
        self.landing_duration.copy_(torch.where(landing, duration, torch.zeros_like(duration)))
        valid = (landing & (duration >= self.cfg.side_swing_min_s - 1e-6)
                 & (duration <= self.cfg.side_air_timeout_s + 1e-6)
                 & (self.peak_gap >= self.cfg.side_swing_clearance_m))
        landing_value = ((duration - self.cfg.side_swing_min_s) /
                         (self.cfg.side_swing_full_reward_s - self.cfg.side_swing_min_s)).clamp(0., 1.)
        step_reward = (landing_value * valid).sum(dim=1)

        self.armed |= self.stable_contact
        self.unloaded_time.masked_fill_(self.stable_contact, 0.)
        self.had_air &= ~self.stable_contact
        self.peak_gap.masked_fill_(self.stable_contact, 0.)
        # A one-frame tap never resets this clock; never-landed feet also accrue cost.
        excess = ((self.unloaded_time - self.cfg.side_air_timeout_s) /
                  self.cfg.side_air_timeout_s).clamp_min(0.)
        air_cost = excess.square().clamp_max(self.cfg.side_air_cost_cap).sum(dim=1) * active
        return load_cost, air_cost, step_reward
