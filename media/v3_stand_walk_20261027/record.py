"""RH-only new V3 update 800: standing and forward, 8 seconds each."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import isaacgym
from isaacgym import gymapi, gymtorch
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LStandWalkRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner

ROOT = Path('/home/xiaohengli/Program/robo_competition')
FFMPEG = '/home/xiaohengli/miniconda3/envs/multinex/bin/ffmpeg'
SCHEDULE = [('STAND', [0.,0.,0.]), ('FORWARD', [.35,0.,0.]), ('STOP', [0.,0.,0.]), ('RESTART', [.35,0.,0.])]


class RecordingRobot(LZQ1LStandWalkRobot):
    def create_sim(self):
        self.graphics_device_id = int(os.environ['RECORD_GRAPHICS_DEVICE'])
        super().create_sim()

    def _resample_commands(self, ids):
        self.commands[ids] = 0.
        self.command_sample_mode[ids] = 0
        self._clear_support(ids)
        self.v2_air_steps[ids] = 0
        self._clear_gait_state(ids)

    def _reset_root_states(self, ids):
        self.root_states[ids] = self.base_init_state
        self.root_states[ids, :3] += self.env_origins[ids]
        self.root_states[ids, 7:13] = 0.
        indices = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                   gymtorch.unwrap_tensor(indices), len(indices))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, required=True)
    opts, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    out = opts.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected = json.loads((ROOT/'media/v3_stand_walk_20261027/selected.json').read_text())
    run = Path(selected['run_dir'])
    checkpoint = run/'model_800.pt'
    assert args.task == 'lz_q1l_flat_stand_walk'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == selected['checkpoint_sha256']
    training_metadata = json.loads((run/'run_metadata.json').read_text())
    cfg, train = task_registry.get_cfgs(args.task)
    archived=json.loads((run/'env_config.json').read_text())
    archived['domain_rand'].pop('push_interval',None)
    runtime_cfg=class_to_dict(cfg);runtime_cfg.pop('inference')
    assert archived == runtime_cfg, 'Gait physics/rewards differ from training archive'
    source_name='HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py'
    archive_sources=list((run/'provenance').rglob('*lz_q1l_2.py'))
    assert len(archive_sources)==1
    def class_tree(path,name):
        return ast.dump(next(c for c in ast.parse(path.read_text()).body if isinstance(c,ast.ClassDef) and c.name==name),include_attributes=False)
    for name in ['LZQ1LV1SupportRobot','LZQ1LV2UnifiedRobot','LZQ1LV2GaitRobot']:
        assert class_tree(ROOT/source_name,name)==class_tree(archive_sources[0],name),name
    for source in ['HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
                   'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
                   'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf']:
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==training_metadata['code_sha256'][source]
    for source, expected in training_metadata['code_sha256'].items():
        if source.endswith(('/envs/__init__.py','/lz_q1l_2.py','/lz_q1l_2_config.py')):
            original=(run/'provenance'/source).read_bytes()
            assert hashlib.sha256(original).hexdigest()==expected
            assert (ROOT/source).read_bytes().startswith(original),source
        else:
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == expected, source
    args.num_envs = cfg.env.num_envs = 1
    args.headless = True
    cfg.env.episode_length_s = 40.
    cfg.commands.resampling_time = 1e6
    cfg.noise.add_noise = False
    cfg.sim.use_gpu_pipeline = cfg.sim.physx.use_gpu = True
    cfg.sim.physx.num_threads = 2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    task_registry.task_classes[args.task] = RecordingRobot
    env = encoder = None
    try:
        env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
        runner = HIMOnPolicyRunner(env, class_to_dict(train), log_dir=None, device=env.device)
        saved = torch.load(checkpoint, map_location=env.device)
        assert saved['iter'] == 800
        runner.alg.actor_critic.load_state_dict(saved['model_state_dict'], strict=True)
        baseline = torch.load(training_metadata['teacher_checkpoint'], map_location='cpu')['model_state_dict']
        frozen = [k for k in baseline if k == 'std' or k.startswith(('actor.', 'estimator.'))]
        assert frozen and all(torch.equal(saved['model_state_dict'][k].cpu(), baseline[k]) for k in frozen)
        assert any(k.startswith('right_rear_delta.') for k in saved['model_state_dict'])
        print('CHECKPOINT_800_FROZEN_BASE_VERIFIED', len(frozen), flush=True)
        policy = runner.get_inference_policy(device=env.device)
        assert policy.policy_mode == 'left-mirror+baseline-stand-model800-forward'
        props = gymapi.CameraProperties()
        props.width, props.height = 640, 480
        cameras = [env.gym.create_camera_sensor(env.envs[0], props) for _ in range(2)]
        assert all(c >= 0 for c in cameras)
        names = env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])
        feet = [names[i] for i in env.feet_indices.tolist()]
        font='fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:fontsize=19:fontcolor=white'
        filters=['drawbox=x=0:y=0:w=iw:h=94:color=black@0.75:t=fill',
                 "drawtext=%s:x=16:y=38:text='RIGHT SIDE VIEW | dt=1ms | policy=50Hz'"%font,
                 "drawtext=%s:x=656:y=38:text='REAR RIGHT VIEW'"%font,
                 "drawtext=%s:x=16:y=66:text='Continuous sequence | baseline STAND / model800 WALK | 0.3s RH blend'"%font]
        for i,(name,cmd) in enumerate(SCHEDULE):
            label='BASELINE / MODEL800 HYBRID | ' + '%s | vx=%.2f vy=%.2f m/s yaw=%.2f rad/s'%(name,*cmd)
            filters.append("drawtext=%s:x=16:y=8:text='%s':enable='gte(t,%d)*lt(t,%d)'"%(font,label,i*8,(i+1)*8))
        video=out/'hybrid_stand_walk_transition.mp4'
        encoder_env=os.environ.copy();encoder_env.pop('LD_LIBRARY_PATH',None)
        encoder=subprocess.Popen([FFMPEG,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x480','-r','25','-i','pipe:0','-vf',','.join(filters),'-an','-c:v','libx264',
            '-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
            stdin=subprocess.PIPE,env=encoder_env)
        trace,resets,segments=[],[],[]
        frames=0
        with torch.no_grad():
            obs,_=env.reset()
            for step in range(100):
                env.commands.zero_();obs[:,:3]=0.
                obs,_,_,done,_,_,_=env.step(policy(obs))
                if bool(done[0]):resets.append(dict(segment=-1,phase='preroll',step=step))
            for segment,(name,command) in enumerate(SCHEDULE):
                first_reset=400
                samples=[]
                for step in range(400):
                    env.commands.zero_();env.commands[0,:3]=torch.tensor(command,device=env.device)
                    env.command_sample_mode[0]=int(command[0]>0)
                    obs[:,:3]=env.commands[:,:3]*env.commands_scale
                    mirrored=bool(policy.mirror_mask(obs)[0])
                    assert not mirrored
                    obs,_,_,done,_,_,_=env.step(policy(obs))
                    reset=bool(done[0])
                    if reset:
                        first_reset=min(first_reset,step)
                        resets.append(dict(segment=segment,phase='video',step=step,time_s=segment*8+(step+1)*.02))
                    row=dict(segment=segment,time_s=segment*8+(step+1)*.02,command=command,mirrored=mirrored,
                        velocity=env.base_lin_vel[0].cpu().tolist(),yaw_rate=float(env.base_ang_vel[0,2]),
                        position=env.root_states[0,:3].cpu().tolist(),
                        quaternion_xyzw=env.base_quat[0].cpu().tolist(),
                        alpha=float(env._stand_walk_actor.alpha[0]),joint_target_deg=torch.rad2deg(env.joint_pos_target[0]).cpu().tolist(),
                        contact_force_z=env.contact_forces[0,env.feet_indices,2].cpu().tolist(),
                        gap_mm=(env.support_foot_gap[0]*1000).cpu().tolist(),reset=reset)
                    trace.append(row);samples.append(row)
                    if step%2==0:
                        p=env.root_states[0,:3].cpu().numpy()
                        for camera,offset in zip(cameras,[(.05,-1.15,.36),(-.85,-.95,.42)]):
                            env.gym.set_camera_location(camera,env.envs[0],gymapi.Vec3(p[0]+offset[0],p[1]+offset[1],offset[2]),gymapi.Vec3(p[0],p[1],.18))
                        env.gym.fetch_results(env.sim,True)
                        env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
                        images=[env.gym.get_camera_image(env.sim,env.envs[0],c,gymapi.IMAGE_COLOR).reshape(480,640,4)[:,:,:3].copy() for c in cameras]
                        encoder.stdin.write(np.concatenate(images,axis=1).tobytes());frames+=1
                valid=samples[50:first_reset]
                summary=dict(name=name,command=command,mirrored=False,valid_samples=len(valid),first_reset_step=first_reset if first_reset<400 else None)
                if valid:
                    summary.update(mean_velocity=np.mean([r['velocity'] for r in valid],axis=0).tolist(),
                        mean_yaw_rate=float(np.mean([r['yaw_rate'] for r in valid])),
                        contact_rate=(np.array([r['contact_force_z'] for r in valid])>1).mean(0).tolist())
                segments.append(summary)
                print('SEGMENT_COMPLETE',name,json.dumps(summary),flush=True)
        encoder.stdin.close()
        assert encoder.wait(timeout=60)==0 and frames==800
        assert len(trace)==1600 and np.isfinite([r['velocity'] for r in trace]).all()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        sources=[Path(__file__).resolve()]+[ROOT/p for p in training_metadata['code_sha256'] if (ROOT/p).exists()]
        sources += [ROOT/'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py', ROOT/'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_stand_walk.py']
        (out/'provenance').mkdir()
        for i,source in enumerate(sources):shutil.copyfile(source,out/'provenance'/('%02d_'%i+source.name))
        meta=dict(variant='baseline_stand_model800_walk_blend_0.3s',task=args.task,checkpoint=str(checkpoint),checkpoint_sha256=digest,iteration=800,
            duration_s=32,fps=25,frames=frames,feet=feet,segments=segments,resets=resets,
            gpu=os.environ['RECORD_GPU_UUID'],graphics_device=os.environ['RECORD_GRAPHICS_DEVICE'],
            controller='Baseline standing, model800 forward, 0.3s RH-only blend; left mirror retained',
            protocol='One nominal reset, 2s baseline stand preroll; continuous 8s stand, 8s forward, 8s stop, 8s restart. No resets at command changes. Noise/domain randomization disabled. No failures removed.',
            limitations='Single nominal trial per action, not a robustness evaluation.',
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        for name,data in [('metadata',meta),('trace',trace),('config',class_to_dict(cfg))]:
            (out/(name+'.json')).write_text(json.dumps(data,indent=2 if name!='trace' else None,allow_nan=False)+'\n')
        print('RESULT',video,flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            try:encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:encoder.kill();encoder.wait()
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
