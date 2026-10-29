"""Sanity check of the LZ-Q1L asset: zero actions (PD holds default pose) on flat ground."""
import isaacgym  # noqa: F401  (must be imported before torch)
import torch

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args()
    args.headless = True
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 16
    env_cfg.noise.add_noise = False
    for flag in ["randomize_friction", "push_robots", "disturbance", "randomize_payload_mass",
                 "randomize_com_displacement", "randomize_kp", "randomize_kd", "delay"]:
        setattr(env_cfg.domain_rand, flag, False)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
    print("dof_names:", env.dof_names)
    print("bodies:", [(n, round(p.mass, 3)) for n, p in zip(body_names, props)])
    print("total mass: %.3f kg" % sum(p.mass for p in props))
    print("feet:", [body_names[i] for i in env.feet_indices.tolist()])
    print("penalised:", [body_names[i] for i in env.penalised_contact_indices.tolist()])
    print("termination:", [body_names[i] for i in env.termination_contact_indices.tolist()])
    print("torque_limits:", env.torque_limits.tolist())
    print("dof_pos_limits:", env.dof_pos_limits.tolist())

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    steps = int(4.0 / env.dt)
    for i in range(steps):
        env.step(actions)
        if i % 25 == 0 or i == steps - 1:
            fz = env.contact_forces[:, env.feet_indices, 2]
            print("t=%.2fs base_z=%.4f grav_xy=%.4f |tau|max=%.2f foot_fz=%s resets=%d" % (
                (i + 1) * env.dt, env.root_states[:, 2].mean().item(),
                env.projected_gravity[:, :2].norm(dim=1).max().item(),
                env.torques.abs().max().item(),
                [round(v, 1) for v in fz.mean(0).tolist()],
                int(env.reset_buf.sum().item())))
    print("foot z (world):", [round(v, 4) for v in env.feet_pos[:, :, 2].mean(0).tolist()])
    print("dof_pos:", [round(v, 3) for v in env.dof_pos[0].tolist()])
    print("torques:", [round(v, 2) for v in env.torques[0].tolist()])


if __name__ == "__main__":
    main()
