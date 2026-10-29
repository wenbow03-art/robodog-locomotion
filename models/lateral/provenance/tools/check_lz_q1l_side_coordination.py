"""Review-only invariants, recorded-trajectory calibration and disposable GPU check."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.lz_q1l_side_coordination import SideCoordinationTracker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'media/side_coordination_review_20261029'
TASK = 'lz_q1l_flat_side_coordination'


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def diff(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [x for k in sorted(set(a)|set(b)) for x in diff(a.get(k),b.get(k),prefix+'.'+k)]
    return [] if a == b else [prefix]


def same(a, b):
    if isinstance(a, torch.Tensor):return torch.equal(a,b)
    if isinstance(a, dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a, (list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a == b


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');opts=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    cfg, train=task_registry.get_cfgs(TASK);old,_=task_registry.get_cfgs('lz_q1l_flat_native_side_height1ms')
    changes=diff(class_to_dict(old),class_to_dict(cfg))
    assert all(x.startswith('.rewards.side_') or x in ('.rewards.scales.lateral_load_floor','.rewards.scales.lateral_air_timeout') for x in changes)
    assert cfg.sim.dt==.001 and cfg.control.decimation==20
    assert cfg.rewards.scales.base_height==-100 and cfg.rewards.scales.tracking_lateral==3
    assert cfg.rewards.scales.feet_air_time==.3
    source=ROOT/'HIMLoco/legged_gym'/train.runner.source_checkpoint
    assert hashlib.sha256(source.read_bytes()).hexdigest()==train.runner.source_sha256
    write('config_audit.json',dict(task=TASK,changes=changes,env=class_to_dict(cfg),train=class_to_dict(train),source=str(source),source_sha256=train.runner.source_sha256))
    c=cfg.rewards;dt=.02;active=torch.ones(2,dtype=torch.bool)
    def tracker():return SideCoordinationTracker(2,'cpu',dt,c)
    force=torch.full((2,4),40.);gap=torch.zeros(2,4)
    t=tracker()
    for _ in range(60):v=t.update(force,gap,active)
    assert all(torch.equal(x,torch.zeros(2)) for x in v)
    # One absent leg: full-window deficit=1, prolonged suspension charged, no landing reward.
    f=force.clone();f[:,1]=0;g=gap.clone();g[:,1]=.02
    for _ in range(60):v=t.update(f,g,active)
    assert torch.allclose(v[0],torch.ones(2),atol=1e-5) and (v[1]>0).all() and (v[2]==0).all()
    # 20 ms tap does not reset the timer; 40 ms support does, but late landing earns zero.
    before=t.unloaded_time[:,1].clone();v=t.update(force,gap,active)
    assert (t.unloaded_time[:,1]>before).all() and (v[2]==0).all()
    v=t.update(force,gap,active);assert (t.unloaded_time[:,1]==0).all() and (v[2]==0).all()
    def landing_reward(steps,clearance):
        x=tracker()
        for _ in range(3):x.update(force,gap,active)
        reward=0.
        gg=gap.clone();gg[:,1]=clearance
        for _ in range(steps):reward+=float(x.update(f,gg,active)[2][0])
        for _ in range(5):reward+=float(x.update(force,gap,active)[2][0])
        return reward
    values={str(k):landing_reward(k,.02) for k in (1,5,8,12,15,16,30)}
    assert values['1']==0 and 0<values['5']<values['8']==values['12']==values['15']==1.
    assert values['16']==values['30']==0 and landing_reward(8,.005)==0
    # First landing after initialization is not a step, even with clearance.
    x=tracker()
    for _ in range(8):x.update(f,g,active)
    assert float(x.update(force,gap,active)[2].sum())==0
    assert float(x.update(force,gap,active)[2].sum())==0
    # Partial reset leaves the other environment intact; inactive commands clear all clocks/history.
    before=t.history[:,1].clone();t.reset(torch.tensor([0]))
    assert t.samples[0]==0 and t.samples[1]==50 and torch.equal(t.history[:,1],before)
    z=t.update(force,gap,torch.zeros(2,dtype=torch.bool));assert all((q==0).all() for q in z)
    assert (t.unloaded_time==0).all() and (t.history==0).all() and not t.armed.any()
    # Mirror/permutation symmetry: all feet share identical rules.
    a=tracker();b=tracker();perm=torch.tensor([1,0,3,2])
    for k in range(70):
        ff=f if k%20<12 else force;gg=g if k%20<12 else gap
        va=a.update(ff,gg,active);vb=b.update(ff[:,perm],gg[:,perm],active)
        assert all(torch.allclose(xx,yy,atol=1e-6) for xx,yy in zip(va,vb))
    # Normal alternating support can earn reward with zero long-window deficit and no overstay.
    x=tracker();earned=0.
    for k in range(100):
        ff=force.clone();gg=gap.clone();air=[0,3] if k%20<8 else ([1,2] if 10<=k%20<18 else [])
        ff[:,air]=0.;gg[:,air]=.02;v=x.update(ff,gg,active);earned+=float(v[2].sum())
        assert (v[1]<1e-6).all()
        if k>=50:assert (v[0]==0).all()
    assert earned>0
    cross_mode=tracker()
    with torch.inference_mode():cross_mode.update(force,gap,active)
    cross_mode.reset(torch.tensor([0]))
    assert cross_mode.samples[0]==0
    write('synthetic_checks.json',dict(passed=True,landing_reward_by_air_steps=values,checks=['balanced support','absent foot','tap debounce','bounded duration reward','minimum clearance','no initial landing bonus','subset reset','inactive reset','permutation symmetry','alternating gait']))
    # Feed the recorded native trajectories through the exact new tracker (no policy update).
    result=json.loads((ROOT/'media/side_height1ms300_native_20261029/latest_result.json').read_text())
    trace=json.loads((Path(result['video']).parent/'trace.json').read_text());cal=[]
    for segment in (0,1):
        t=SideCoordinationTracker(1,'cpu',dt,c);rows=[];air=torch.zeros(1,4);last=torch.zeros(1,4,dtype=torch.bool)
        for row in trace[segment*250:(segment+1)*250]:
            ff=torch.tensor([row['contact_force_z']]);gg=torch.tensor([row['gap_mm']])/1000
            new=t.update(ff,gg,torch.ones(1,dtype=torch.bool))
            contact=ff>1;filtered=contact|last;first=(air>0)&filtered;last=contact
            air+=dt;old_air=((air-.3)*first).sum();air*=~filtered
            rows.append([float(new[0])*c.scales.lateral_load_floor,float(new[1])*c.scales.lateral_air_timeout,float(new[2])*c.scales.feet_air_time,float(old_air)*c.scales.feet_air_time])
        rows=rows[50:];cal.append(dict(direction='left' if segment==0 else 'right',weighted_rate_mean=[sum(x[j] for x in rows)/len(rows) for j in range(4)],weighted_rate_max_magnitude=[max(abs(x[j]) for x in rows) for j in range(4)]))
    write('trace_calibration.json',dict(columns=['load_floor','air_timeout','bounded_landing','old_air_time'],rows=cal,units='Reward rate scale before multiplication by policy dt=.02; landing terms occur only at events',scope='Frozen native trace replay, no evidence of learning improvement'))
    print('CPU_AND_TRACE_PASS',json.dumps(cal),flush=True)
    if not opts.gpu:return
    from legged_gym.utils import get_args
    from rsl_rl.runners.him_side_coordination_runner import HIMSideCoordinationRunner
    sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0','--num_envs','32']
    args=get_args();cfg.env.num_envs=32;env=None
    try:
        env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMSideCoordinationRunner(env,class_to_dict(train),log_dir=None,device=env.device)
        model=runner.alg.actor_critic;saved=torch.load(source,map_location=env.device)
        assert same(saved['model_state_dict'],model.state_dict())
        assert same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
        assert same(saved['estimator_optimizer_state_dict'],model.estimator.optimizer.state_dict())
        estimator={k:v.clone() for k,v in model.estimator.state_dict().items()};actor={k:v.clone() for k,v in model.actor.state_dict().items()}
        assert not hasattr(runner.alg,'teacher') and all(not p.requires_grad for p in model.estimator.parameters())
        obs,_=env.reset();crit=env.get_privileged_observations()
        assert torch.equal(env.build_inference_policy(model)(obs),model.act_inference(obs))
        for _ in range(100):
            with torch.inference_mode():
                action=runner.alg.act(obs,crit);obs,crit,reward,done,infos,ids,terminal=env.step(action)
                nxt=crit.clone();nxt[ids]=terminal;runner.alg.process_env_step(reward,done,infos,nxt)
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all() and torch.isfinite(env.side_values).all()
            assert int((env.commands[:,1]>0).sum())==16 and int((env.commands[:,1]<0).sum())==16
        with torch.inference_mode():runner.alg.compute_returns(crit)
        losses=runner.alg.update();assert all(math.isfinite(x) for x in losses)
        assert same(estimator,model.estimator.state_dict()) and not same(actor,model.actor.state_dict())
        # Repeated reward reads cannot advance the new trackers.
        state=env.side_tracker.unloaded_time.clone()
        assert torch.equal(env._reward_lateral_load_floor(),env._reward_lateral_load_floor())
        assert torch.equal(state,env.side_tracker.unloaded_time)
        # Reset clears only requested environments. Resampling also clears their windows.
        # Isaac Gym's base reset_buf is created inside rollout inference mode.
        # Exercise lifecycle hooks in that same context, as the training loop does.
        with torch.inference_mode():
            env.reset_idx(torch.tensor([0],device=env.device));assert env.side_tracker.samples[0]==0
            env._resample_commands(torch.tensor([1],device=env.device));assert env.side_tracker.samples[1]==0
            env.episode_length_buf.fill_(1)
            env.commands[2,1]*=-1
            env.commands[3,:3]=0.
            env._post_physics_step_callback()
            assert env.side_tracker.samples[2]==1 and env.side_tracker.samples[3]==0
            assert not env.side_active[3] and (env.side_values[3]==0).all()
            assert env._reward_feet_air_time()[3]==0
        write('gpu_checks.json',dict(passed=True,envs=32,steps=100,disposable_updates=1,no_checkpoint_saved=True,teacher_absent=True,native_actions=True,estimator_unchanged=True,actor_updated=True,optimizer_restored=True,metrics=env.pop_side_coordination_metrics()))
        print('GPU_PREFLIGHT_PASS',flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
