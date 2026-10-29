"""Frozen composite: accepted stand/walk baseline + legacy V8 sidestepping."""
import torch
from .lz_q1l_stand_walk import StandWalkResidualActor
from .lz_q1l_mirror import LZQ1LMirror, LeftMirroredLZQ1LPolicy


class LegacyLateralPolicy:
    policy_mode = 'baseline-stand-walk+v8-lateral-left-mirror-1ms'

    def __init__(self, baseline, lateral, env):
        cfg = env.cfg.inference
        self.baseline = StandWalkResidualActor(baseline, env.dt, cfg.blend_time_s,
            cfg.stand_threshold, cfg.forward_threshold, cfg.other_command_tolerance)
        self.lateral = LeftMirroredLZQ1LPolicy(lateral,
            LZQ1LMirror.from_env(env), env.commands_scale)
        self.command_scale = env.commands_scale

    def lateral_mask(self, history):
        c = history[:, :3] / self.command_scale
        return (c[:, 1].abs() > .05) & (c[:, [0, 2]].abs() <= 1e-6).all(dim=1)

    def mirror_mask(self, history):
        return self.lateral_mask(history) & self.lateral.mirror_mask(history)

    def reset(self, ids=None):
        self.baseline.reset(ids)

    @torch.no_grad()
    def act_inference(self, history):
        result = self.baseline.act_inference(history)
        mask = self.lateral_mask(history)
        # Both experts consume the physical history, including actually applied
        # actions. Only V8's left branch reflects all six frames internally.
        if mask.any():
            result[mask] = self.lateral(history[mask])
        return result

    def __call__(self, history):
        return self.act_inference(history)

    @torch.no_grad()
    def estimator(self, history):
        velocity, latent = self.baseline.estimator(history)
        mask = self.lateral_mask(history)
        if mask.any():
            v, z = self.lateral.estimator(history[mask])
            velocity = velocity.clone(); latent = latent.clone()
            velocity[mask] = v; latent[mask] = z
        return velocity, latent
