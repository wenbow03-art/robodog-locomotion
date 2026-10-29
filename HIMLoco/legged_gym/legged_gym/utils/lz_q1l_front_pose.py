"""Rolling mean front-joint posture cost; angles in radians."""
import math
import torch

class FrontPoseTracker:
    def __init__(self, n, device, dt, window_s=1., deadband_deg=10., scale_deg=10.):
        self.steps = round(window_s / dt)
        self.deadband = math.radians(deadband_deg)
        self.scale = math.radians(scale_deg)
        self.history = torch.zeros(self.steps, n, 2, 2, device=device)
        self.total = torch.zeros(n, 2, 2, device=device)
        self.samples = torch.zeros(n, dtype=torch.long, device=device)
        self.mean = torch.zeros_like(self.total)
        self.cost = torch.zeros(n, device=device)
        self.cursor = 0
    def reset(self, ids):
        self.history[:, ids] = 0.
        self.total[ids] = 0.
        self.samples[ids] = 0
        self.mean[ids] = 0.
        self.cost[ids] = 0.
    def update(self, angles, active):
        self.reset((~active).nonzero(as_tuple=False).flatten())
        old = self.history[self.cursor]
        self.total.sub_(old)
        old.copy_(angles * active[:, None, None])
        self.total.add_(old)
        self.samples.add_(active.long()).clamp_(max=self.steps)
        self.mean.copy_(self.total / self.samples.clamp_min(1)[:, None, None])
        difference = (self.mean[:, 0] - self.mean[:, 1]).abs()
        self.cost.copy_(((difference - self.deadband).clamp_min(0.) / self.scale).square().sum(1)
                        * (active & (self.samples == self.steps)))
        self.cursor = (self.cursor + 1) % self.steps
        return self.cost
