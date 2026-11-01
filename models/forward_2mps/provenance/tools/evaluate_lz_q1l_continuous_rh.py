"""Frozen nominal start-to-2m/s gate; all trials, including failures, are retained."""
import argparse
import sys
from pathlib import Path
import isaacgym
from isaacgym import gymtorch
import torch
import numpy as np
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LForwardAccelerationRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules.him_stance_width_actor import HIMStanceWidthActorCritic
from lz_q1l_continuous_rh_common import TASK, attach_standing, digest, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stage', type=int, required=True, choices=[0,1,2])
    p.add_argument('--out', type=Path, required=True)
    opts = p.parse_args()
    opts.out.mkdir(parents=True, exist_ok=False)
    import json
    checksum = digest(opts.checkpoint)
    base=Path(__file__).resolve().parents[1]
    meta=json.loads((opts.checkpoint.parent/'run_metadata.json').read_text())
    assert meta['task']==TASK
    for f,h in meta['code_sha256'].items():assert digest(base/f)==h,f
    cfg, train = task_registry.get_cfgs(TASK)
    c = cfg.acceleration
    levels = c.levels[:opts.stage+1]
    trials = c.evaluation_trials
    n = trials*len(levels)
    cfg.env.num_envs = n
    cfg.env.episode_length_s = c.evaluation_seconds+2
    cfg.noise.add_noise = False
    cfg.sim.physx.num_threads = 2
    for k in dir(cfg.domain_rand):
        if not k.startswith('_') and isinstance(getattr(cfg.domain_rand,k), bool):
            setattr(cfg.domain_rand,k,False)
    starts = np.random.RandomState(417).uniform(-.03,.03,(trials,6));starts[0] = 0.
    class EvaluationRobot(LZQ1LForwardAccelerationRobot):
        def _resample_commands(self, ids):
            super()._resample_commands(ids)
            self.ramp_target[ids] = c.high_speed
            self.ramp_rate[ids] = torch.tensor(levels,device=self.device)[ids//trials]
        def _reset_root_states(self, ids):
            self.root_states[ids] = self.base_init_state
            self.root_states[ids,:3] += self.env_origins[ids]
            self.root_states[ids,7:13] = torch.tensor(starts,dtype=torch.float,device=self.device)[ids%trials]
            ix=ids.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(ix),len(ix))
    task_registry.task_classes[TASK] = EvaluationRobot
    sys.argv = [sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
    args = get_args();env=None
    try:
        env,_ = task_registry.make_env(TASK,args=args,env_cfg=cfg)
        model = HIMStanceWidthActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,env.num_actions,**class_to_dict(train.policy)).to(env.device)
        model.configure_right_rear(env.dof_names,env.commands_scale)
        model.load_state_dict(torch.load(opts.checkpoint,map_location=env.device)['model_state_dict'],strict=True)
        model.eval().requires_grad_(False)
        policy = env.build_inference_policy(model)
        attach_standing(env)
        assert model.continuous_rh_forward
        rh=model.right_rear_indices
        frames=[];dones=[];targets=[];joints=[]
        with torch.inference_mode():
            obs,_=env.reset()
            for _ in range(round(c.evaluation_seconds/env.dt)):
                obs,_,_,done,_,_,_=env.step(policy(obs))
                targets.append(env.joint_pos_target[:,rh].cpu().numpy().copy())
                joints.append(env.dof_pos[:,rh].cpu().numpy().copy())
                frames.append(env.ramp_step_metrics.cpu().numpy().copy())
                dones.append(done.cpu().numpy().copy())
        trace=np.asarray(frames);failed=np.asarray(dones).astype(bool)
        groups=[]
        for g,a in enumerate(levels):
            rows=[]
            for j in range(trials):
                ix=g*trials+j;t=trace[:,ix];active=t[:,7]>c.standing_seconds
                reset_times=(np.flatnonzero(failed[:,ix])+1)*env.dt
                row=dict(start=j,passed=False,reset_times_s=reset_times.tolist())
                if not len(reset_times):
                    # Every no-reset trial remains visible even if slow, curved, or stalled.
                    speed=t[:,1];band=(speed>=c.speed_band[0])&(speed<=c.speed_band[1])&active
                    count=0;arrival=None
                    for k,ok in enumerate(band):
                        count=count+1 if ok else 0
                        if count>=round(c.speed_hold_seconds/env.dt):
                            arrival=float(t[k,7]-c.standing_seconds);break
                    mean=float(speed[-round(5/env.dt):].mean())
                    heading=float(np.rad2deg(np.abs(t[active,5])).max())
                    lateral=float(np.abs(t[active,6]).max())
                    row.update(actual_vx=mean,arrival_and_hold_seconds=arrival,max_heading_deg=heading,
                               max_lateral_m=lateral,mean_abs_yaw_rate=float(np.abs(t[active,3]).mean()))
                    row['passed']=bool(c.speed_band[0]<=mean<=c.speed_band[1] and arrival is not None
                                       and arrival<=c.high_speed/a+c.arrival_slack_seconds
                                       and heading<=c.max_heading_degrees and lateral<=c.max_lateral_metres)
                rows.append(row)
            groups.append(dict(acceleration=a,trials=rows,passed=sum(r['passed'] for r in rows)))
        assert digest(opts.checkpoint)==checksum
        np.savez_compressed(opts.out/'trace.npz',metrics=trace,done=failed)
        report=dict(checkpoint=str(opts.checkpoint.resolve()),sha256=checksum,stage=opts.stage,groups=groups,
                    starts=starts.tolist(),criteria=class_to_dict(c),
                    protocol='Frozen/no noise/no domain randomization;20 matched velocity perturbations per level;original standing2s+full-action0.3s blend+ramp to2m/s;16s;failures retained. Nominal gate, not robustness certification.')
        assert opts.stage==0 and n==20
        reference_path=base/'media/acceleration300_continuous_gate_20261031'
        ref=Path((reference_path/'latest_run.txt').read_text().strip())/'result'
        assert json.loads((ref/'report.json').read_text())['starts']==starts.tolist()
        ref_m=np.load(ref/'trace.npz')['metrics']
        ref_j=np.load(ref/'joint_trace.npz')
        ref_names=json.loads((ref/'joints.json').read_text())['dof_names']
        assert ref_names==list(env.dof_names)
        ix=rh.cpu().tolist()
        q=np.asarray(joints);tar=np.asarray(targets);limits=env.dof_pos_limits[rh].cpu().numpy()
        def trial_summary(m,done,q,target):
            vals=[]
            for i in range(20):
                sw=int(np.flatnonzero(m[:,i,0]>0)[0]);resets=np.flatnonzero(done[:,i])
                end=int(resets[0]) if len(resets) else len(m)
                k=min(sw+20,end)
                if k<=sw:
                    vals.append(dict(reset=True)); continue
                pitch=q[sw:k,i];yaw=m[min(sw+15,end-1),i,5]
                jump=np.abs(np.rad2deg(np.diff(target[sw-1:k,i],axis=0))).max()
                vals.append(dict(reset=bool(len(resets)),startup_complete=(end>=sw+20),
                    startup_yaw_abs_deg=float(abs(np.rad2deg(yaw))),RH_max_target_jump_deg=float(jump),
                    RH_soft_limit=bool(np.any((pitch<limits[:,0])|(pitch>limits[:,1]))),
                    max_lateral_m=float(np.abs(m[sw:end,i,6]).max()),
                    last5_actual_vx=None if len(resets) else float(m[-250:,i,1].mean())))
            return vals
        rows=trial_summary(trace,failed,q,tar)
        prior=trial_summary(ref_m,np.zeros(ref_m.shape[:2],bool),ref_j['q'][:,:,ix],ref_j['target'][:,:,ix])
        def summary(v):
            full=[r for r in v if not r['reset']]; early=[r for r in v if r.get('startup_complete')]
            def mean(rr,k):return float(np.mean([r[k] for r in rr])) if rr else None
            return dict(no_reset=len(full),speed_in_band_and_no_reset=sum(1.9<=r['last5_actual_vx']<=2.1 for r in full),
                startup_complete=len(early),RH_soft_limit_trials=sum(r['RH_soft_limit'] for r in early),
                mean_abs_yaw_at_0p3s_deg=mean(early,'startup_yaw_abs_deg'),
                max_RH_startup_target_jump_deg=max((r['RH_max_target_jump_deg'] for r in early),default=None),
                survivor_mean_max_lateral_m=mean(full,'max_lateral_m'),survivor_mean_last5_vx=mean(full,'last5_actual_vx'))
        report['comparison']=dict(control=summary(prior),checkpoint=summary(rows),trials=rows,
            paired_lateral_improved=sum(not r['reset'] and r['max_lateral_m']<prior[i]['max_lateral_m'] for i,r in enumerate(rows)),
            note='Relative to the untrained continuous gate, same starts. Failed trials cannot count as improved. Startup window .4s, sample yaw at .3s.')
        np.savez_compressed(opts.out/'rh_startup.npz',q=q,target=tar,soft_limits=limits)
        write(opts.out/'comparison.json',report['comparison'])
        write(opts.out/'report.json',report)
        print('EVALUATION_COMPLETE',[(r['acceleration'],r['passed']) for r in groups],flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
