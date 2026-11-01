"""Heading-aligned foot widths and short rolling mean; no force/angle equality."""
import torch

def heading_widths(centers, root_position, quaternion):
    delta = centers - root_position[:, None, :]
    x, y, z, w = quaternion.unbind(-1)
    yaw = torch.atan2(2. * (w*z + x*y), 1. - 2. * (y*y + z*z))
    lateral = -yaw.sin()[:, None]*delta[:, :, 0] + yaw.cos()[:, None]*delta[:, :, 1]
    return torch.stack(((lateral[:, 2]-lateral[:, 3]).abs(),
                        (lateral[:, 0]-lateral[:, 1]).abs()), dim=1)

class StanceWidthTracker:
    def __init__(self, n, device, dt, window_s=.2, scale_m=.1):
        self.steps = max(1, round(window_s / dt))
        self.scale = scale_m
        self.history = torch.zeros(self.steps, n, 2, device=device)
        self.total = torch.zeros(n, 2, device=device)
        self.mean = torch.zeros_like(self.total)
        self.samples = torch.zeros(n, dtype=torch.long, device=device)
        self.cost = torch.zeros(n, device=device)
        self.cursor = 0
    def reset(self, ids):
        self.history[:, ids] = 0.
        self.total[ids] = 0.
        self.mean[ids] = 0.
        self.samples[ids] = 0
        self.cost[ids] = 0.
    def update(self, widths, active):
        self.reset((~active).nonzero(as_tuple=False).flatten())
        old = self.history[self.cursor]
        self.total.sub_(old)
        old.copy_(widths * active[:, None])
        self.total.add_(old)
        self.samples.add_(active.long()).clamp_(max=self.steps)
        self.mean.copy_(self.total / self.samples.clamp_min(1)[:, None])
        self.cost.copy_(((self.mean[:, 0]-self.mean[:, 1])/self.scale).square()
                        * (active & (self.samples == self.steps)))
        self.cursor = (self.cursor + 1) % self.steps
        return self.cost
