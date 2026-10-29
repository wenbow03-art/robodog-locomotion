"""Per-foot landing periods, with geometric and contact debounce validation."""
import math
import torch


class FootCycleTracker:
    def __init__(self, n, device, dt, cfg):
        self.dt, self.cfg = dt, cfg
        self.air_steps = max(1, math.ceil(cfg.lateral_cycle_min_air_s / dt))
        self.contact_steps = max(1, math.ceil(cfg.lateral_cycle_contact_s / dt))
        self.age = torch.zeros(n, 4, device=device)
        self.air = torch.zeros(n, 4, dtype=torch.long, device=device)
        self.contact = torch.zeros_like(self.air)
        self.peak = torch.zeros_like(self.age)
        self.grounded = torch.zeros_like(self.age, dtype=torch.bool)
        self.armed = torch.zeros_like(self.grounded)
        self.seen = torch.zeros_like(self.grounded)
        self.last_error = torch.zeros_like(self.age)

    def reset(self, ids):
        for value in (self.age, self.air, self.contact, self.peak, self.grounded,
                      self.armed, self.seen, self.last_error):
            value[ids] = 0

    def update(self, force, gap, active):
        cfg = self.cfg
        self.reset((~active).nonzero(as_tuple=False).flatten())
        self.age += active[:, None] * self.dt
        touching = (force >= cfg.lateral_cycle_contact_n) & active[:, None]
        flying = (force <= cfg.lateral_cycle_air_n) & active[:, None]
        self.contact = torch.where(touching, self.contact + 1, torch.zeros_like(self.contact))
        self.air = torch.where(flying, self.air + 1, torch.zeros_like(self.air))
        self.peak = torch.where(flying, torch.maximum(self.peak, gap), self.peak)
        self.armed |= (self.grounded & (self.air >= self.air_steps)
                       & (self.peak >= cfg.lateral_cycle_min_clearance_m))
        stable = self.contact >= self.contact_steps
        landing = self.armed & stable
        measured = landing & self.seen
        period = torch.where(measured, self.age, torch.zeros_like(self.age))
        error = ((self.age - cfg.lateral_cycle_target_s).abs()
                 - cfg.lateral_cycle_tolerance_s).clamp_min(0.) / cfg.lateral_cycle_target_s
        self.last_error = torch.where(measured, error.clamp_max(2.).square(), self.last_error)
        self.seen |= landing
        self.age = torch.where(landing, torch.zeros_like(self.age), self.age)
        self.armed &= ~landing
        self.peak = torch.where(stable, torch.zeros_like(self.peak), self.peak)
        self.grounded |= stable
        overdue = ((self.age - cfg.lateral_cycle_target_s - cfg.lateral_cycle_tolerance_s)
                   / cfg.lateral_cycle_target_s).clamp(0., 2.).square()
        # Persist the last completed-cycle error between landings; not a sparse
        # one-timestep penalty. Overdue grows even if a foot never takes a step.
        penalty = torch.maximum(overdue, self.last_error).mean(-1) * active
        return penalty, measured, period, landing
