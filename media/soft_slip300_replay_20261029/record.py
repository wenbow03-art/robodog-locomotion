"""Record native side model300 side commands, both5s, no learning."""
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
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LSoftSlipRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import quat_apply
import xml.etree.ElementTree as ET
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules.him_actor_critic import HIMActorCritic
from types import SimpleNamespace

ROOT = Path('/home/xiaohengli/Program/robo_competition')
FFMPEG = '/home/xiaohengli/miniconda3/envs/multinex/bin/ffmpeg'
SCHEDULE = [('LEFT', [0.,.3,0.]), ('RIGHT', [0.,-.3,0.])]


class RecordingRobot(LZQ1LSoftSlipRobot):
    def _init_buffers(self):
        super()._init_buffers()
        names=self.gym.get_actor_rigid_body_names(self.envs[0],self.actor_handles[0])
        xml=ET.parse(self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)).getroot()
        offsets=[];radii=[]
        for i in self.feet_indices.tolist():
            cs=xml.findall("link[@name='%s']/collision"%names[i]);assert len(cs)==1
            c=cs[0];origin=c.find('origin')
            offsets.append([float(x) for x in origin.get('xyz','0 0 0').split()] if origin is not None else [0.,0.,0.])
            radii.append(float(c.find('geometry/sphere').get('radius')))
        self.support_sphere_offsets=torch.tensor(offsets,device=self.device).expand(self.num_envs,-1,-1)
        self.support_sphere_radii=torch.tensor(radii,device=self.device)
        self.support_foot_gap=torch.zeros(self.num_envs,4,device=self.device)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        body=self.rigid_body_states.view(self.num_envs,self.num_bodies,13)[:,self.feet_indices]
        offset=quat_apply(body[:,:,3:7].reshape(-1,4),self.support_sphere_offsets.reshape(-1,3)).reshape(self.num_envs,4,3)
        self.support_foot_gap=body[:,:,2]+offset[:,:,2]-self.support_sphere_radii

    def create_sim(self):
        self.graphics_device_id = int(os.environ['RECORD_GRAPHICS_DEVICE'])
        super().create_sim()

    def _resample_commands(self, ids):
        self.commands[ids] = 0.

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
    parser.add_argument('--physics-ms', type=int, choices=[1], default=1)
    opts, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = get_args()
    out = opts.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assert args.task == 'lz_q1l_flat_soft_slip'
    selected=json.loads((Path(__file__).parent/'selected.json').read_text())
    metadata=json.loads((Path(selected['run_dir'])/'run_metadata.json').read_text())
    for name,sha in metadata['code_sha256'].items():
        current=(ROOT/name).read_bytes()
        current_sha=hashlib.sha256(current).hexdigest()
        if current_sha!=sha:
            audit=selected['append_only_source_audit'][name]
            archived=(Path(selected['run_dir'])/'provenance'/name).read_bytes()
            assert hashlib.sha256(archived).hexdigest()==sha==audit['training_sha256']
            assert current.startswith(archived) and current[len(archived):].decode()==audit['suffix']
            assert current_sha==audit['current_sha256']
        else:assert current_sha==sha
    assert opts.physics_ms==1
    cfg, train = task_registry.get_cfgs(args.task)
    assert class_to_dict(cfg)==json.loads((Path(selected['run_dir'])/'env_config.json').read_text())
    assert cfg.sim.dt==.001 and cfg.control.decimation==20
    args.num_envs = cfg.env.num_envs = 1
    args.headless = True
    cfg.env.episode_length_s = 60.
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
        checkpoint=Path(selected['checkpoint'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==selected['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location=env.device);assert saved['iter']==300
        model=HIMActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,env.num_actions,**class_to_dict(train)['policy']).to(env.device)
        model.load_state_dict(saved['model_state_dict'],strict=True);model.eval().requires_grad_(False)
        policy=env.build_inference_policy(model)
        runner=SimpleNamespace(sources={'soft_slip300':dict(path=str(checkpoint),sha256=selected['checkpoint_sha256'],iteration=300)})
        assert abs(env.dt-.02)<1e-9
        props = gymapi.CameraProperties()
        props.width, props.height = 640, 480
        cameras = [env.gym.create_camera_sensor(env.envs[0], props) for _ in range(2)]
        assert all(c >= 0 for c in cameras)
        names = env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])
        feet = [names[i] for i in env.feet_indices.tolist()]
        font='fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:fontsize=19:fontcolor=white'
        filters=['drawbox=x=0:y=0:w=iw:h=94:color=black@0.75:t=fill',
                 "drawtext=%s:x=16:y=38:text='RIGHT SIDE VIEW | physics=%dms | policy=50Hz'"%(font,opts.physics_ms),
                 "drawtext=%s:x=656:y=38:text='REAR RIGHT VIEW'"%font,
                 "drawtext=%s:x=16:y=66:text='HEIGHT -100 | tracking reward=3.0 | NO MIRROR | model300'"%font]
        for i,(name,cmd) in enumerate(SCHEDULE):
            label='SOFT SLIP -0.05 | model300 | ' + '%s | vx=%.2f vy=%.2f m/s yaw=%.2f rad/s'%(name,*cmd)
            filters.append("drawtext=%s:x=16:y=8:text='%s':enable='gte(t,%d)*lt(t,%d)'"%(font,label,i*5,(i+1)*5))
        video=out/'soft_slip300_gait_lateral.mp4'
        encoder_env=os.environ.copy();encoder_env.pop('LD_LIBRARY_PATH',None)
        encoder=subprocess.Popen([FFMPEG,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x480','-r','25','-i','pipe:0','-vf',','.join(filters),'-an','-c:v','libx264',
            '-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],
            stdin=subprocess.PIPE,env=encoder_env)
        trace,resets,segments=[],[],[]
        frames=0
        def checked_action(obs):
            result = policy(obs)
            expected = model.act_inference(obs)
            assert torch.equal(result, expected)
            assert torch.isfinite(result).all()
            return result
        with torch.no_grad():
            obs,_=env.reset()
            for step in range(100):
                env.commands.zero_();obs[:,:3]=0.
                obs,_,_,done,_,_,_=env.step(policy(obs))
                if bool(done[0]):resets.append(dict(segment=-1,phase='preroll',step=step))
            for segment,(name,command) in enumerate(SCHEDULE):
                first_reset=250
                samples=[]
                for step in range(250):
                    env.commands.zero_();env.commands[0,:3]=torch.tensor(command,device=env.device)
                    obs[:,:3]=env.commands[:,:3]*env.commands_scale
                    mirrored=False
                    obs,_,_,done,_,_,_=env.step(checked_action(obs))
                    reset=bool(done[0])
                    if reset:
                        first_reset=min(first_reset,step)
                        resets.append(dict(segment=segment,phase='video',step=step,time_s=segment*5+(step+1)*.02))
                    body=env.rigid_body_states.view(env.num_envs,env.num_bodies,13)[:,env.feet_indices]
                    offset=quat_apply(body[:,:,3:7].reshape(-1,4),env.support_sphere_offsets.reshape(-1,3)).reshape(1,4,3)
                    center=body[:,:,:3]+offset
                    bottom=offset.clone();bottom[:,:,2]-=env.support_sphere_radii
                    bottom_velocity=body[:,:,7:10]+torch.cross(body[:,:,10:13],bottom,dim=-1)
                    center_velocity=body[:,:,7:10]+torch.cross(body[:,:,10:13],offset,dim=-1)
                    row=dict(foot_center_world=center[0].cpu().tolist(),
                        foot_bottom_velocity_world=bottom_velocity[0].cpu().tolist(),
                        foot_center_velocity_world=center_velocity[0].cpu().tolist(),
                        segment=segment,time_s=segment*5+(step+1)*.02,command=command,mirrored=mirrored,
                        velocity=env.base_lin_vel[0].cpu().tolist(),yaw_rate=float(env.base_ang_vel[0,2]),
                        position=env.root_states[0,:3].cpu().tolist(),
                        quaternion_xyzw=env.base_quat[0].cpu().tolist(),
                        joint_actual_deg=torch.rad2deg(env.dof_pos[0]).cpu().tolist(),
                        joint_target_deg=torch.rad2deg(env.joint_pos_target[0]).cpu().tolist(),
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
                summary=dict(name=name,command=command,mirrored=False,valid_samples=len(valid),first_reset_step=first_reset if first_reset<250 else None)
                if valid:
                    summary.update(mean_velocity=np.mean([r['velocity'] for r in valid],axis=0).tolist(),
                        mean_yaw_rate=float(np.mean([r['yaw_rate'] for r in valid])),
                        contact_rate=(np.array([r['contact_force_z'] for r in valid])>1).mean(0).tolist())
                segments.append(summary)
                print('SEGMENT_COMPLETE',name,json.dumps(summary),flush=True)
        encoder.stdin.close()
        assert encoder.wait(timeout=60)==0 and frames==250
        assert len(trace)==500 and np.isfinite([r['velocity'] for r in trace]).all()
        for source in runner.sources.values():
            assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
        sources = [Path(__file__).resolve()] + [ROOT/p for p in metadata['code_sha256']]
        (out/'provenance').mkdir()
        for i,source in enumerate(sources):shutil.copyfile(source,out/'provenance'/('%02d_'%i+source.name))
        meta=dict(variant='both_native_no_mirror',task=args.task,sources=runner.sources,
            duration_s=10,fps=25,frames=frames,feet=feet,joint_names=env.dof_names,segments=segments,resets=resets,
            gpu=os.environ['RECORD_GPU_UUID'],graphics_device=os.environ['RECORD_GRAPHICS_DEVICE'],
            strict_checkpoint_loaded=True,append_only_source_audit=selected['append_only_source_audit'],
            physics_dt=cfg.sim.dt,decimation=cfg.control.decimation,
            protocol='2s native side model300 stand preroll; LEFT .30 then RIGHT -.30, each5s. Both directions native, no mirroring; no stage resets. Native side model300 for side commands and standing warmup. Noise/randomization off; Evaluation disables noise and domain randomization including action delay; trained1ms/decimation20 preserved.',
            limitations='Isolated native side model300 test with its own standing warmup, not a baseline stand-to-side transition test. Single nominal trial, no robustness claim.',
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
