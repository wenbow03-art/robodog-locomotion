"""Paired frozen-policy evaluation: posture500 versus gait200, 24 trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import quat_apply
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from check_lz_q1l_v2_gait import Probe, STARTS, COMMANDS, ROOT
from train_lz_q1l_v1_support_500 import differences


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out',type=Path,required=True)
    opts,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest;args=get_args()
    out=opts.out;out.mkdir(parents=True,exist_ok=False)
    current=Path(json.loads((ROOT/'media/v2_gait_20261025/latest_training.json').read_text())['run_dir'])
    source=Path(json.loads((current/'run_metadata.json').read_text())['source_checkpoint'])
    checkpoints=[source,current/'model_200.pt'];labels=['posture500','gait200']
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints]
    assert hashes==['ed10805f5778f24b1c249dbe254460473acb8ac39d85c9f85142fc2b93816e1f',
                    'fab8580057253836256d4d41743176070dd3fca9c0cf4aa718871f5d337d85a2']
    cfg,train=task_registry.get_cfgs('lz_q1l_flat_v2_gait')
    prior=json.loads((source.parent/'env_config.json').read_text());prior['domain_rand'].pop('push_interval',None)
    diff=differences(prior,class_to_dict(cfg));assert all(k.startswith('rewards.') for k in diff)
    cfg.env.num_envs=args.num_envs=24;cfg.env.episode_length_s=20.
    cfg.commands.resampling_time=1e6;cfg.noise.add_noise=False
    cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=True;cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    args.headless=True;task_registry.task_classes[args.task]=Probe
    manifest=dict(labels=labels,checkpoints=[str(p) for p in checkpoints],checkpoint_sha256=hashes,
        starts=STARTS.tolist(),commands=COMMANDS,backend=args.sim_device,training_updates=0,
        protocol='Each policy: 4 commands x 3 paired starts. Stand 2s then command 8s. Statistics command seconds 1-8, before first reset. No noise/domain randomization.',
        physics_diff=diff,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'tools/check_lz_q1l_v2_gait.py',ROOT/'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py']})
    save(out/'manifest.json',manifest);save(out/'config.json',class_to_dict(cfg))
    env=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        policies=[]
        for path in checkpoints:
            runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device=env.device)
            runner.alg.actor_critic.load_state_dict(torch.load(path,map_location=env.device)['model_state_dict'],strict=True)
            policies.append(runner.get_inference_policy(device=env.device))
        obs,_=env.reset();alive=torch.ones(24,dtype=torch.bool,device=env.device);resets=[];data={}
        with torch.no_grad():
            for step in range(500):
                env.commands.zero_();env.command_sample_mode.zero_()
                if step>=100:
                    for model in range(2):
                        for c,(_,speed) in enumerate(COMMANDS):
                            ids=slice(model*12+c*3,model*12+c*3+3)
                            env.commands[ids,0]=speed;env.command_sample_mode[ids]=int(speed>0)
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                actions=torch.cat([policies[0](obs[:12]),policies[1](obs[12:])])
                obs,_,reward,done,_,_,_=env.step(actions)
                for i in (alive&done.bool()).nonzero().flatten().tolist():resets.append(dict(index=i,time_s=(step+1)*.02))
                alive &= ~done.bool()
                body=env.rigid_body_states.view(24,env.num_bodies,13)[:,env.feet_indices]
                offset=quat_apply(body[:,:,3:7].reshape(-1,4),env.support_sphere_offsets.reshape(-1,3)).reshape(24,4,3)
                offset[:,:,2]-=env.support_sphere_radii
                bottom=body[:,:,7:10]+torch.cross(body[:,:,10:13],offset,dim=-1)
                vals=dict(valid=alive,fz=env.contact_forces[:,env.feet_indices,2],gap=env.support_foot_gap,
                    velocity=env.base_lin_vel,angular_velocity=env.base_ang_vel,height=env.root_states[:,2],
                    slip=bottom[:,:,:2].norm(dim=-1),terms=env.gait_penalties)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                for key,value in vals.items():data.setdefault(key,[]).append(value.cpu().numpy().copy())
                if (step+1)%100==0:print('STEP',step+1,flush=True)
        arrays={k:np.asarray(v) for k,v in data.items()};np.savez_compressed(out/'traces.npz',**arrays)
        trials=[]
        for i in range(24):
            speed=COMMANDS[(i%12)//3][1];mask=arrays['valid'][150:,i]
            row=dict(model=labels[i//12],command=speed,start=i%3,failed=any(r['index']==i for r in resets),valid_samples=int(mask.sum()))
            if mask.any():
                a={k:v[150:,i][mask] for k,v in arrays.items()};f=a['fz'].clip(0);g=a['gap']*1000;v=a['velocity'];contact=f>1
                forces=f.mean(0)
                row.update(vx=float(v[:,0].mean()),vx_mae=float(np.abs(v[:,0]-speed).mean()),
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
