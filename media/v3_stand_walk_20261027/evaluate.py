"""Continuous stand/walk/stop/restart, baseline vs model800 vs hybrid; no updates."""
from pathlib import Path
import json,hashlib,sys,os
import isaacgym
import torch
import numpy as np
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LStandWalkRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from check_lz_q1l_v2_gait import Probe as GaitProbe
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
SCHEDULE=[('stand',0.),('walk',.35),('stop',0.),('restart',.35)]
class Probe(LZQ1LStandWalkRobot):
    _reset_root_states=GaitProbe._reset_root_states
    _resample_commands=GaitProbe._resample_commands

def main():
    checks=json.loads((P/'policy_checks.json').read_text());assert all(checks['checks'].values())
    ck=Path(checks['checkpoint']);assert hashlib.sha256(ck.read_bytes()).hexdigest()==checks['checkpoint_sha256']
    out=P/'evaluation';out.mkdir(exist_ok=False)
    args=get_args();assert args.task=='lz_q1l_flat_stand_walk'
    cfg,train=task_registry.get_cfgs(args.task);args.num_envs=cfg.env.num_envs=9;cfg.env.episode_length_s=40.;cfg.noise.add_noise=False;cfg.commands.resampling_time=1e6;cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    task_registry.task_classes[args.task]=Probe;env=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg);runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device=env.device);model=runner.alg.actor_critic
        saved=torch.load(ck,map_location=env.device);assert saved['iter']==800;model.load_state_dict(saved['model_state_dict'],strict=True)
        policy=runner.get_inference_policy(device=env.device);obs,_=env.reset();data={};resets=[];alive=torch.ones(9,dtype=torch.bool,device=env.device)
        with torch.no_grad():
            for step in range(1700):
                segment=-1 if step<100 else (step-100)//400;speed=0. if segment<0 else SCHEDULE[segment][1]
                env.commands.zero_();env.commands[:,0]=speed;env.command_sample_mode[:]=int(speed>0);obs[:,:3]=env.commands[:,:3]*env.commands_scale
                actions=policy(obs);raw=model.act_inference(obs);v,z=model.estimator(obs);base=model.actor(torch.cat((obs[:,:45],v,z),-1))
                if step>=100 and (step-100)%400>=15:
                    expected=base if speed==0 else raw
                    assert torch.equal(actions[6:],expected[6:]),(step,'endpoint mismatch')
                other=torch.tensor([0,1,2,6,7,8,9,10,11],device=env.device);assert torch.equal(actions[:,other],base[:,other])
                actions[:3]=base[:3];actions[3:6]=raw[3:6]
                alpha=env._stand_walk_actor.alpha.clone();alpha[:3]=0;alpha[3:6]=1
                obs,_,reward,done,_,_,_=env.step(actions)
                for i in (alive&done.bool()).nonzero().flatten().tolist():resets.append(dict(index=i,time_s=(step+1)*.02-2))
                alive &= ~done.bool()
                values=dict(valid=alive,fz=env.contact_forces[:,env.feet_indices,2],gap=env.support_foot_gap,quat=env.base_quat,height=env.root_states[:,2],velocity=env.base_lin_vel,angular_velocity=env.base_ang_vel,joint_target=env.joint_pos_target,alpha=alpha)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                for k,v in values.items():data.setdefault(k,[]).append(v.cpu().numpy().copy())
                if (step+1)%400==0:print('STEP',step+1,flush=True)
        arr={k:np.asarray(v) for k,v in data.items()};np.savez_compressed(out/'traces.npz',**arr)
        trials=[]
        for mi,label in enumerate(['baseline','model800','hybrid']):
            for stage,(name,speed) in enumerate(SCHEDULE):
                for si in range(3):
                    ix=mi*3+si;lo=100+stage*400;start=lo+150;end=lo+400;mask=arr['valid'][start:end,ix]
                    row=dict(controller=label,stage=name,start=si,valid_samples=int(mask.sum()),failed=any(r['index']==ix for r in resets))
                    if mask.any():
                        f=arr['fz'][start:end,ix][mask].clip(0);force=f.mean(0);q=arr['quat'][start:end,ix][mask];x,y,z,w=q.T;roll=np.rad2deg(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
                        row.update(force_n=force.tolist(),rear_gap_n=float(abs(force[0]-force[1])),front_gap_n=float(abs(force[2]-force[3])),left_share_pct=float(100*(force[0]+force[2])/force.sum()),roll_abs_deg=float(abs(roll).mean()),height_cm=float(arr['height'][start:end,ix][mask].mean()*100),vx_mae=float(abs(arr['velocity'][start:end,ix][mask,0]-speed).mean()),contact_pct=((f>1).mean(0)*100).tolist(),all_four_contact_pct=float((f>1).all(1).mean()*100))
                    begin=max(lo-1,0);window=slice(begin,lo+50);q=arr['quat'][window,ix];x,y,z,w=q.T;roll=np.rad2deg(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
                    row.update(transition_roll_max_deg=float(abs(roll).max()),transition_rh_target_jump_max_deg=np.rad2deg(abs(np.diff(arr['joint_target'][window,ix,3:6],axis=0))).max(0).tolist(),transition_height_min_cm=float(arr['height'][window,ix].min()*100))
                    trials.append(row)
        groups=[]
        for label in ['baseline','model800','hybrid']:
            for name,_ in SCHEDULE:
                selected=[r for r in trials if r['controller']==label and r['stage']==name];keys=[k for k in selected[0] if k not in ['controller','stage','start','failed']]
                groups.append(dict(controller=label,stage=name,**{k:dict(mean=np.mean([r[k] for r in selected],axis=0).tolist(),std=np.std([r[k] for r in selected],axis=0).tolist()) for k in keys}))
        report=dict(groups=groups,trials=trials,resets=resets,protocol='9 environments, 3 matched root-velocity starts per controller. One reset then2s preroll and continuous stand8/walk8/stop8/restart8. Steady stats last5s of each segment; transition stats first1s including preceding command sample. No hidden resets/noise/randomization. 50Hz. Three starts are not training seeds.',checks=dict(endpoint_outputs_exact=True,other_nine_outputs_exact=True),checkpoint=str(ck),checkpoint_sha256=checks['checkpoint_sha256'],gpu=os.environ['CUDA_VISIBLE_DEVICES'])
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'config.json').write_text(json.dumps(class_to_dict(cfg),indent=2)+'\n')
        assert hashlib.sha256(ck.read_bytes()).hexdigest()==checks['checkpoint_sha256']
        print('COMPLETE',json.dumps(dict(resets=resets,groups=groups)),flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)
if __name__=='__main__':main()
