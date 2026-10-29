"""Compare source upright150 and all stand/front checkpoints with paired initial states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import quat_apply
from isaacgym import gymtorch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from check_lz_q1l_v2_gait import Probe as GaitProbe, STARTS, COMMANDS, ROOT
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2StandFrontRobot


class Probe(LZQ1LV2StandFrontRobot):
    _reset_root_states = GaitProbe._reset_root_states

    def _reset_dofs(self, ids):
        factors=np.stack([np.ones(self.num_dof),np.random.RandomState(11).uniform(.5,1.5,self.num_dof),np.random.RandomState(12).uniform(.5,1.5,self.num_dof)])
        self.dof_pos[ids]=self.default_dof_pos*torch.as_tensor(factors,dtype=torch.float,device=self.device)[ids%3]
        self.dof_vel[ids]=0.
        ix=ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.dof_state),gymtorch.unwrap_tensor(ix),len(ix))

    def _resample_commands(self, ids):
        GaitProbe._resample_commands(self, ids)
        self._clear_stand_force(ids)

from train_lz_q1l_v1_support_500 import differences


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out',type=Path,required=True)
    opts,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest;args=get_args()
    out=opts.out;out.mkdir(parents=True,exist_ok=False)
    current=Path(json.loads((ROOT/'media/v2_stand_front_20261025/latest_training.json').read_text())['run_dir'])
    source=Path(json.loads((current/'run_metadata.json').read_text())['source_checkpoint'])
    updates=[150,300]
    checkpoints=[source]+[current/('model_%d.pt'%i) for i in updates[1:]]
    labels=['upright150','standfront300']
    num_envs=len(checkpoints)*12
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints]
    assert hashes[0]=='301538cc20d262f548357af854d8b2338c1955c626c7ef9d9207d361439001d0'
    assert hashes[-1]=='0061682614802f1e3ceeae16f6f0dd6a147604534a0db004a9a2297b56ed6509'
    cfg,train=task_registry.get_cfgs('lz_q1l_flat_v2_stand_front')
    prior=json.loads((source.parent/'env_config.json').read_text());prior['domain_rand'].pop('push_interval',None)
    diff=differences(prior,class_to_dict(cfg));assert all(k.startswith('rewards.') for k in diff)
    for checkpoint in checkpoints:
        archived=json.loads((checkpoint.parent/'env_config.json').read_text());archived['domain_rand'].pop('push_interval',None)
        assert all(k.startswith('rewards.') for k in differences(archived,class_to_dict(cfg)))
    cfg.env.num_envs=args.num_envs=num_envs;cfg.env.episode_length_s=20.
    cfg.commands.resampling_time=1e6;cfg.noise.add_noise=False
    cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=True;cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    args.headless=True;task_registry.task_classes[args.task]=Probe
    manifest=dict(additional_updates=updates,labels=labels,checkpoints=[str(p) for p in checkpoints],checkpoint_sha256=hashes,
        starts=STARTS.tolist(),commands=COMMANDS,backend=args.sim_device,training_updates=0,
        protocol='Each policy: 4 commands x 3 paired starts. Paired root velocities and joint poses (nominal plus RandomState 11/12 joint factors .5-1.5). Stand 2s then command 8s. Statistics command seconds 1-8, before first reset. No noise/domain randomization.',
        physics_diff=diff,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'tools/check_lz_q1l_v2_gait.py',ROOT/'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',ROOT/'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py']})
    save(out/'manifest.json',manifest);save(out/'config.json',class_to_dict(cfg))
    env=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        policies=[]
        for index,path in enumerate(checkpoints):
            runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device=env.device)
            state=torch.load(path,map_location=env.device)
            assert state['iter']==updates[index]
            assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
            runner.alg.actor_critic.load_state_dict(state['model_state_dict'],strict=True)
            policies.append(runner.get_inference_policy(device=env.device))
        obs,_=env.reset();alive=torch.ones(num_envs,dtype=torch.bool,device=env.device);resets=[];data={}
        with torch.no_grad():
            for step in range(500):
                env.commands.zero_();env.command_sample_mode.zero_()
                if step>=100:
                    for model in range(len(checkpoints)):
                        for c,(_,speed) in enumerate(COMMANDS):
                            ids=slice(model*12+c*3,model*12+c*3+3)
                            env.commands[ids,0]=speed;env.command_sample_mode[ids]=int(speed>0)
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                actions=torch.cat([policy(obs[i*12:(i+1)*12]) for i,policy in enumerate(policies)])
                obs,_,reward,done,_,_,_=env.step(actions)
                for i in (alive&done.bool()).nonzero().flatten().tolist():resets.append(dict(index=i,time_s=(step+1)*.02))
                alive &= ~done.bool()
                body=env.rigid_body_states.view(num_envs,env.num_bodies,13)[:,env.feet_indices]
                offset=quat_apply(body[:,:,3:7].reshape(-1,4),env.support_sphere_offsets.reshape(-1,3)).reshape(num_envs,4,3)
                offset[:,:,2]-=env.support_sphere_radii
                bottom=body[:,:,7:10]+torch.cross(body[:,:,10:13],offset,dim=-1)
                vals=dict(valid=alive,fz=env.contact_forces[:,env.feet_indices,2],gap=env.support_foot_gap,
                    velocity=env.base_lin_vel,angular_velocity=env.base_ang_vel,height=env.root_states[:,2],
                    slip=bottom[:,:,:2].norm(dim=-1),terms=env.gait_penalties,
                    standing_terms=env.stand_force_penalties,quat=env.base_quat,front_term=env.front_force_penalty,
                    stand_extra=torch.stack((env._reward_stand_height_extra(),env._reward_stand_orientation_extra()),dim=-1))
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                for key,value in vals.items():data.setdefault(key,[]).append(value.cpu().numpy().copy())
                if (step+1)%100==0:print('STEP',step+1,flush=True)
        arrays={k:np.asarray(v) for k,v in data.items()};np.savez_compressed(out/'traces.npz',**arrays)
        trials=[]
        for i in range(num_envs):
            speed=COMMANDS[(i%12)//3][1];mask=arrays['valid'][150:,i]
            row=dict(additional_update=updates[i//12],model=labels[i//12],command=speed,start=i%3,failed=any(r['index']==i for r in resets),valid_samples=int(mask.sum()))
            if mask.any():
                a={k:v[150:,i][mask] for k,v in arrays.items()};f=a['fz'].clip(0);g=a['gap']*1000;v=a['velocity'];contact=f>1
                forces=f.mean(0)
                q=a['quat'];x,y,z,w=q.T
                roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))*180/np.pi
                pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))*180/np.pi
                row.update(front_balance_raw=float(a['front_term'].mean()),stand_extra_raw=a['stand_extra'].mean(0).tolist(),roll_mean_deg=float(roll.mean()),pitch_mean_deg=float(pitch.mean()),
                    roll_abs_mean_deg=float(np.abs(roll).mean()),pitch_abs_mean_deg=float(np.abs(pitch).mean()),roll_std_deg=float(roll.std()),pitch_std_deg=float(pitch.std()),
                    height_mae_cm=float(np.abs(a['height']-.30).mean()*100),
                    vy_mae=float(np.abs(v[:,1]).mean()),yaw_mae=float(np.abs(a['angular_velocity'][:,2]).mean()),
                    front_force_difference_n=float(abs(forces[2]-forces[3])),
                    standing_penalties=a['standing_terms'].mean(0).tolist(),vx=float(v[:,0].mean()),vx_mae=float(np.abs(v[:,0]-speed).mean()),
                    height_cm=float(a['height'].mean()*100),height_std_mm=float(a['height'].std()*1000),
                    contact_pct=(contact.mean(0)*100).tolist(),gap_p95_mm=np.percentile(g,95,axis=0).tolist(),gap_max_mm=g.max(0).tolist(),
                    force_n=forces.tolist(),rear_force_difference_n=float(abs(forces[0]-forces[1])),
                    rear_imbalance=float(abs(forces[0]-forces[1])/(forces[0]+forces[1]+1)),
                    contact_slip_m_s=[float(a['slip'][:,j][contact[:,j]].mean()) if contact[:,j].any() else None for j in range(4)],
                    loaded_slip_m_s=[float(a['slip'][:,j][f[:,j]>10].mean()) if (f[:,j]>10).any() else None for j in range(4)],
                    raw_penalties=a['terms'].mean(0).tolist())
            trials.append(row)
        assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints]
        save(out/'report.json',dict(trials=trials,resets=resets,checkpoint_weights_unchanged=True,
            feet=['LH','RH','LF','RF'],limitations='Three paired initial states, one terrain, one trained checkpoint per method; not training-seed uncertainty. Slip is rigid sphere-bottom velocity estimate.'))
        print('COMPLETE',out,flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
