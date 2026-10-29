"""Paired normal/mirrored-forward diagnostic with 1ms PD telemetry; no learning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2UnifiedRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror
from rsl_rl.runners import HIMOnPolicyRunner
from diagnose_no_policy_support import STARTS

ROOT = Path(__file__).resolve().parents[1]
B = len(STARTS)

def save(p, data): p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

class TrackingRobot(LZQ1LV2UnifiedRobot):
    def _resample_commands(self, ids):
        self.commands[ids] = 0
        self.command_sample_mode[ids] = 0
        self._clear_support(ids)
        self.v2_air_steps[ids] = 0

    def _reset_root_states(self, ids):
        self.root_states[ids] = self.base_init_state
        self.root_states[ids, :3] += self.env_origins[ids]
        starts = torch.as_tensor(STARTS, dtype=torch.float, device=self.device)[ids % B].clone()
        # Match the initial state under the same reflection as the diagnostic policy.
        starts[ids >= B] *= starts.new_tensor([1,-1,1,-1,1,-1])
        self.root_states[ids, 7:13] = starts
        ids32 = ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(ids32), len(ids32))

    def _compute_torques(self, actions):
        torque = super()._compute_torques(actions)
        if hasattr(self, 'probe'):
            raw = self.p_gains*self.Kp_factors*(self.joint_pos_target-self.dof_pos)-self.d_gains*self.Kd_factors*self.dof_vel
            self.probe.append(torch.stack((self.dof_pos,self.dof_vel,self.joint_pos_target,raw,torque),1).detach().cpu().numpy().copy())
        return torque

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out',type=Path,required=True)
    opts,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest;args=get_args()
    assert args.task=='lz_q1l_flat_v2_posture'
    out=opts.out.resolve();out.mkdir(parents=True,exist_ok=False)
    training=Path(json.loads((ROOT/'media/v2_posture_20261024/latest_training.json').read_text())['run_dir'])
    checkpoint=training/'model_500.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    training_meta=json.loads((training/'run_metadata.json').read_text())
    cfg,train=task_registry.get_cfgs(args.task)
    assert cfg.sim.dt==.001 and cfg.control.decimation==20
    cfg.env.num_envs=args.num_envs=2*B;cfg.env.episode_length_s=20.
    cfg.commands.resampling_time=1e6;cfg.noise.add_noise=False
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    args.headless=True
    cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=args.sim_device.startswith('cuda')
    cfg.sim.physx.num_threads=2
    task_registry.task_classes[args.task]=TrackingRobot
    sources=[Path(__file__).resolve(),ROOT/'tools/diagnose_no_policy_support.py'] + [ROOT/p for p in (
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
        'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
        'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
        'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
        'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py',
        'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
        'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf')]
    (out/'provenance').mkdir()
    for i,p in enumerate(sources):
        rel=str(p.relative_to(ROOT));h=hashlib.sha256(p.read_bytes()).hexdigest()
        if rel in training_meta['code_sha256']:assert training_meta['code_sha256'][rel]==h,rel
        shutil.copyfile(p,out/'provenance'/('%02d_'%i+p.name))
    manifest=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,block_size=B,
        modes=['normal_default_forward','diagnostic_full_left_right_reflection'],starts=STARTS.tolist(),
        protocol='Each mode: nominal + 8 small velocity starts (mirrored starts in reflected mode). 2s learned standing then 8s vx=0.35. No randomization. Statistics forward seconds 1-8, excluding any first-reset and later samples. Full reflection is diagnostic only, not a default controller change.',
        physics_dt_s=.001,control_dt_s=.02,backend=args.sim_device,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        question='Compare target/actual joint excursions and tracking errors, torque clipping, geometry, and whether asymmetry follows reflected policy.')
    save(out/'manifest.json',manifest)
    env=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),log_dir=None,device=env.device)
        state=torch.load(checkpoint,map_location=env.device);assert state['iter']==500
        runner.alg.actor_critic.load_state_dict(state['model_state_dict'],strict=True)
        policy=runner.get_inference_policy(device=env.device)
        mirror=LZQ1LMirror.from_env(env)
        names=env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])
        manifest.update(feet=[names[i] for i in env.feet_indices.tolist()],dof_names=env.dof_names,
            torque_limits_Nm=env.torque_limits.cpu().tolist(),velocity_limits_rad_s=env.dof_vel_limits.cpu().tolist(),
            action_clip=float(cfg.normalization.clip_actions))
        save(out/'manifest.json',manifest);save(out/'config.json',class_to_dict(cfg))
        obs,_=env.reset();env.probe=[]
        alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);resets=[]
        data={k:[] for k in ('valid','fz','gap','q','dq','target','action','root','foot_body','foot_base','foot_bottom_velocity','velocity')}
        with torch.no_grad():
            for step in range(500):
                env.commands.zero_();env.command_sample_mode[:]=int(step>=100)
                if step>=100:env.commands[:,0]=.35
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                inputs=obs.clone();inputs[B:]=mirror.history(obs[B:])
                assert not policy.mirror_mask(inputs).any()
                actions=policy(inputs);actions[B:]=mirror.joints(actions[B:])
                obs,_,_,done,_,_,_=env.step(actions)
                for index in (alive & done.bool()).nonzero().flatten().tolist():resets.append(dict(index=index,time_s=(step+1)*.02))
                alive &= ~done.bool()
                body=env.rigid_body_states.view(env.num_envs,env.num_bodies,13)[:,env.feet_indices]
                offset=quat_apply(body[:,:,3:7].reshape(-1,4),env.support_sphere_offsets.reshape(-1,3)).reshape(env.num_envs,4,3)
                foot_center=body[:,:,:3]+offset
                rel=foot_center-env.root_states[:,None,:3]
                foot_base=quat_rotate_inverse(env.base_quat[:,None,:].expand(-1,4,-1).reshape(-1,4),rel.reshape(-1,3)).reshape(env.num_envs,4,3)
                bottom=offset.clone();bottom[:,:,2]-=env.support_sphere_radii
                point_velocity=body[:,:,7:10]+torch.cross(body[:,:,10:13],bottom,dim=-1)
                vals=dict(valid=alive,fz=env.contact_forces[:,env.feet_indices,2],gap=env.support_foot_gap,
                    q=env.dof_pos,dq=env.dof_vel,target=env.joint_pos_target,action=actions,root=env.root_states,
                    foot_body=body,foot_base=foot_base,foot_bottom_velocity=point_velocity,
                    velocity=torch.cat((env.base_lin_vel,env.base_ang_vel),1))
                for k,v in vals.items():data[k].append(v.cpu().numpy().copy())
                if (step+1)%100==0:print('STEP',step+1,flush=True)
        arrays={k:np.asarray(v) for k,v in data.items()};probe=np.asarray(env.probe)
        assert all(np.isfinite(v).all() for v in arrays.values()) and np.isfinite(probe).all()
        assert probe.shape==(10000,18,5,12)
        np.savez_compressed(out/'traces.npz',**arrays)
        np.savez_compressed(out/'substeps.npz',data=probe,dt=.001,columns=['q','dq','target','unclipped_torque','applied_torque'])
        save(out/'completion.json',dict(completed=True,resets=resets,probe_shape=list(probe.shape),
            checkpoint_unchanged=hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest))
        print('COMPLETE',out,'RESETS',resets,flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
