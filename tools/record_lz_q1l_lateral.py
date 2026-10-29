"""Record one lateral direction using the project's default inference policy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import isaacgym  # noqa: F401
from isaacgym import gymapi
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l import LZQ1LRobot
from legged_gym.utils import get_args

FFMPEG = '/home/xiaohengli/miniconda3/envs/multinex/bin/ffmpeg'
WIDTH, HEIGHT = 640, 400


class RecordingRobot(LZQ1LRobot):
    def create_sim(self):
        self.graphics_device_id = self.sim_device_id
        super().create_sim()

    def _resample_commands(self, ids):
        self.commands[ids] = 0.


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--direction', choices=['left', 'right'], required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    opts, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    out = opts.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    video = out / (opts.direction + '.mp4')
    if video.exists():
        raise FileExistsError(video)
    vy = .4 if opts.direction == 'left' else -.4
    schedule = [(2., (0., 0., 0.)), (8., (0., vy, 0.)), (2., (0., 0., 0.))]
    cfg, train_cfg = task_registry.get_cfgs(args.task)
    args.num_envs = cfg.env.num_envs = 1
    args.headless = True
    cfg.env.episode_length_s = 20.
    cfg.noise.add_noise = False
    cfg.commands.curriculum = cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1e6
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    task_registry.task_classes[args.task] = RecordingRobot
    env = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        train_cfg.runner.resume = True
        runner, _ = task_registry.make_alg_runner(env, args.task, args=args, train_cfg=train_cfg)
        policy = runner.get_inference_policy(device=env.device)
        assert policy.policy_mode == 'left-mirror'
        camera_properties = gymapi.CameraProperties()
        camera_properties.width, camera_properties.height = WIDTH, HEIGHT
        camera = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        obs, _ = env.reset()
        trace, frames, resets = [], [], []
        step = 0
        with torch.no_grad():
            for duration, command in schedule:
                for _ in range(round(duration / env.dt)):
                    env.commands[:, :3] = torch.tensor(command, device=env.device)
                    obs[:, :3] = env.commands[:, :3] * env.commands_scale
                    mirrored = bool(policy.mirror_mask(obs)[0])
                    assert mirrored == (command[1] > 0.)
                    obs, _, _, done, _, _, _ = env.step(policy(obs))
                    if bool(done[0]):
                        resets.append(dict(time_s=(step + 1) * env.dt, timeout=bool(env.time_out_buf[0])))
                    trace.append(dict(time_s=(step + 1) * env.dt, command=command,
                                      mirrored=mirrored, reset=bool(done[0]),
                                      velocity=env.base_lin_vel[0].cpu().tolist(),
                                      yaw_rate=float(env.base_ang_vel[0, 2]),
                                      base_z=float(env.root_states[0, 2])))
                    if step % 2 == 0:
                        p = env.root_states[0, :3].cpu().numpy()
                        env.gym.set_camera_location(camera, env.envs[0],
                            gymapi.Vec3(p[0] + .6, p[1] - 1.4, .55),
                            gymapi.Vec3(p[0], p[1], .22))
                        env.gym.fetch_results(env.sim, True)
                        env.gym.step_graphics(env.sim)
                        env.gym.render_all_camera_sensors(env.sim)
                        raw = env.gym.get_camera_image(env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR)
                        frames.append(raw.reshape(HEIGHT, WIDTH, 4)[..., :3].copy())
                    step += 1
                    if step % 100 == 0:
                        print('RECORDED', opts.direction, step, flush=True)
        encoder_env = os.environ.copy()
        encoder_env.pop('LD_LIBRARY_PATH', None)
        branch = 'MIRRORED' if vy > 0 else 'ORIGINAL'
        font = 'fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:fontsize=19:fontcolor=white'
        filters = (
            'drawbox=x=0:y=0:w=iw:h=68:color=black@0.6:t=fill,'
            "drawtext=%s:x=16:y=10:text='V8  %s  |  %s':enable='between(t,2,10)',"
            "drawtext=%s:x=16:y=38:text='command vy = %+.2f m/s':enable='between(t,2,10)',"
            "drawtext=%s:x=16:y=22:text='STAND  |  command = 0':enable='lt(t,2)+gte(t,10)'"
        ) % (font, opts.direction.upper(), branch, font, vy, font)
        subprocess.run([FFMPEG, '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', '%dx%d' % (WIDTH, HEIGHT), '-r', '25', '-i', 'pipe:0',
                        '-vf', filters, '-an', '-c:v', 'libx264', '-crf', '18',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
                       input=np.asarray(frames).tobytes(), env=encoder_env, check=True)
        end = min([r['time_s'] for r in resets] + [10.])
        samples = [r for r in trace if 3. < r['time_s'] < end and r['command'][1] == vy]
        meta = dict(task=args.task, load_run=args.load_run, checkpoint=args.checkpoint,
                    seed=args.seed, direction=opts.direction, policy_mode=policy.policy_mode,
                    schedule=schedule, frames=len(frames), fps=25, duration_s=12,
                    resets=resets, mean_vy_after_1s_before_first_reset=float(np.mean([r['velocity'][1] for r in samples])) if samples else None,
                    limitations='One nominal simulation demonstration; no reliability estimate.')
        video.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
        video.with_suffix('.trace.json').write_text(json.dumps(trace) + '\n')
        print('RECORDED_RESULT', json.dumps(meta), flush=True)
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
