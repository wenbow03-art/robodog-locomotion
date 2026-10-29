"""Sagittal reflection for the LZ-Q1L 45-value observation layout.

Inference only: reflect every history frame, run the unchanged estimator/actor,
then reflect actions back into the physical robot's coordinates. No learned
latent coordinate is assigned a hand-written reflection rule.
"""
import torch


class LZQ1LMirror:
    frame_size = 45

    def __init__(self, dof_names, device="cpu"):
        names = list(dof_names)
        expected = {kind + "_leg_" + leg + "_joint"
                    for kind in ("roll", "pitch01", "pitch02")
                    for leg in ("f_l", "f_r", "b_l", "b_r")}
        if len(names) != 12 or set(names) != expected:
            raise ValueError("Mirror requires the 12 named LZ-Q1L joints")
        partners = [name.replace("_l_joint", "_r_joint") if "_l_joint" in name
                    else name.replace("_r_joint", "_l_joint") for name in names]
        self.permutation = torch.tensor([names.index(name) for name in partners],
                                        dtype=torch.long, device=device)
        self.sign = torch.tensor([-1. if name.startswith("roll_") else 1.
                                  for name in names], device=device)

    @classmethod
    def from_env(cls, env, device=None):
        if env.num_one_step_obs != cls.frame_size or env.num_obs != 6 * cls.frame_size:
            raise ValueError("Expected six 45-value LZ-Q1L observation frames")
        mirror = cls(env.dof_names, env.device)
        if not torch.allclose(mirror.joints(env.default_dof_pos), env.default_dof_pos):
            raise ValueError("Default joint pose must be mirror symmetric")
        # The base PD controller applies its hip multiplier at fixed indices.
        scale = torch.full((12,), float(env.cfg.control.action_scale), device=env.device)
        scale[[0, 3, 6, 9]] *= env.cfg.control.hip_reduction
        if not torch.equal(scale, scale[mirror.permutation]):
            raise ValueError("Action scaling must be mirror symmetric")
        if device is not None:
            mirror.permutation = mirror.permutation.to(device)
            mirror.sign = mirror.sign.to(device)
        return mirror

    def joints(self, values):
        if values.shape[-1] != 12:
            raise ValueError("Expected 12 joints/actions")
        return values.index_select(-1, self.permutation) * self.sign

    def linear_velocity(self, values):
        return values * values.new_tensor([1., -1., 1.])

    def history(self, values):
        if values.shape[-1] != 6 * self.frame_size:
            raise ValueError("Expected six 45-value observation frames")
        frames = values.reshape(*values.shape[:-1], 6, self.frame_size)
        result = frames.clone()
        result[..., :3] = frames[..., :3] * values.new_tensor([1., -1., -1.])
        result[..., 3:6] = frames[..., 3:6] * values.new_tensor([-1., 1., -1.])
        result[..., 6:9] = self.linear_velocity(frames[..., 6:9])
        for start in (9, 21, 33):
            result[..., start:start + 12] = self.joints(frames[..., start:start + 12])
        return result.reshape_as(values)


class LeftMirroredLZQ1LPolicy:
    """Reflect only pure-left commands; all other commands use the original policy.

    Selection uses the newest frame, while all six frames are transformed
    together. Physical histories remain untouched, including across switches.
    The 1e-6 tolerance is in physical command units (m/s and rad/s).
    """

    def __init__(self, actor, mirror, command_scale):
        self.actor = actor
        self.mirror = mirror
        self.policy_mode = 'left-mirror'
        self.command_scale = torch.as_tensor(command_scale, device=mirror.sign.device)
        if self.command_scale.shape != (3,) or not (self.command_scale > 0).all():
            raise ValueError('Expected three positive command normalization scales')

    def mirror_mask(self, history):
        command = history[..., :3] / self.command_scale
        return ((command[..., 1] > 1e-6) & (command[..., 0].abs() <= 1e-6)
                & (command[..., 2].abs() <= 1e-6))

    def selected_history(self, history):
        mask = self.mirror_mask(history)
        selected = torch.where(mask.unsqueeze(-1), self.mirror.history(history), history)
        return selected, mask

    def act_inference(self, history):
        selected, mask = self.selected_history(history)
        actions = self.actor.act_inference(selected)
        return torch.where(mask.unsqueeze(-1), self.mirror.joints(actions), actions)

    def estimator(self, history):
        selected, mask = self.selected_history(history)
        velocity, latent = self.actor.estimator(selected)
        return torch.where(mask.unsqueeze(-1), self.mirror.linear_velocity(velocity), velocity), latent

    def __call__(self, history):
        return self.act_inference(history)
