"""Frozen policies: synchronized joint/force/roll traces and signed roll-rate impulses."""
import argparse,hashlib,json,sys
from pathlib import Path
import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply,quat_rotate_inverse
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from compare_lz_q1l_balanceroll200 import Probe,ROOT
from train_lz_q1l_v1_support_500 import differences

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--out',required=True,type=Path)
    opt,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest;args=get_args();out=opt.out;out.mkdir(parents=True,exist_ok=False)
    run=Path(json.loads((ROOT/'media/v2_balance_roll_20261025/latest_training.json').read_text())['run_dir'])
    source=Path(json.loads((run/'run_metadata.json').read_text())['source_checkpoint'])
    paths=[source,run/'model_200.pt'];hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert hashes==['301538cc20d262f548357af854d8b2338c1955c626c7ef9d9207d361439001d0','b3a921ca0d5eb2c6c16bd113c5300db34bcb7ede49fb541db40bb872097e2d55']
    cfg,train=task_registry.get_cfgs(args.task)
    for path in paths:
        old=json.loads((path.parent/'env_config.json').read_text());old['domain_rand'].pop('push_interval',None)
        assert all(k.startswith('rewards.') for k in differences(old,class_to_dict(cfg)))
    cfg.env.num_envs=args.num_envs=36;cfg.env.episode_length_s=20.;cfg.commands.resampling_time=1e6
    cfg.noise.add_noise=False;cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=True;cfg.sim.physx.num_threads=2
    for k in dir(cfg.domain_rand):
        if not k.startswith('_') and isinstance(getattr(cfg.domain_rand,k),bool):setattr(cfg.domain_rand,k,False)
    args.headless=True;task_registry.task_classes[args.task]=Probe
    cases=[dict(index=m*18+c*9+b*3+s,model=['upright150','balanceroll200'][m],speed=[0.,.35][c],impulse=[-.5,0.,.5][b],start=s) for m in range(2) for c in range(2) for b in range(3) for s in range(3)]
    env=None
    try:
        env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
        assert not env.cfg.domain_rand.delay
        policies=[]
        for path in paths:
            runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device=env.device)
            runner.alg.actor_critic.load_state_dict(torch.load(path,map_location=env.device)['model_state_dict'],strict=True)
            policies.append(runner.get_inference_policy(device=env.device))
        bodies=env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])
        manifest=dict(cases=cases,checkpoints=[str(p) for p in paths],sha256=hashes,dof_names=env.dof_names,
            feet=[bodies[i] for i in env.feet_indices.tolist()],dt=float(env.dt),command_step=100,impulse_step=200,
            protocol='36 environments: 2 models x 2 commands x 3 signed body-roll velocity impulses x 3 matched starts. Commands begin after 2s stand; at 4s add -0.5/0/+0.5 rad/s body-roll rate to root angular velocity. Refresh newest angular-velocity observation only, preserve real history. Record 12s at 50Hz. No training, no noise/randomization. Impulse changes simulated root momentum, not a real push-force model.',
            limitations='Closed-loop timing alone does not establish direction of causation. Parallel branches can differ slightly before impulse; quantify pre-event mismatches. This probes response of a frozen policy, not the cause of training regression.',
            code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'tools/compare_lz_q1l_balanceroll200.py',ROOT/'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py']})
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (out/'config.json').write_text(json.dumps(class_to_dict(cfg),indent=2)+'\n')
        obs,_=env.reset();records={};resets=[]
        def keep(k,v):records.setdefault(k,[]).append(v.detach().cpu().numpy().copy())
        with torch.no_grad():
            for step in range(600):
                env.commands.zero_();env.command_sample_mode.zero_()
                if step>=100:
                    for m in range(2):env.commands[m*18+9:m*18+18,0]=.35;env.command_sample_mode[m*18+9:m*18+18]=1
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                keep('pre_quat',env.base_quat);keep('pre_q',env.dof_pos)
                if step==200:
                    impulse=torch.tensor([c['impulse'] for c in cases],device=env.device)
                    local=torch.zeros(36,3,device=env.device);local[:,0]=impulse
                    env.root_states[:,10:13]+=quat_apply(env.base_quat,local)
                    ids=(impulse!=0).nonzero().flatten().to(torch.int32)
                    env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_states),gymtorch.unwrap_tensor(ids),len(ids))
                    env.base_ang_vel[:]=quat_rotate_inverse(env.base_quat,env.root_states[:,10:13])
                    obs[:,3:6]=env.base_ang_vel*env.obs_scales.ang_vel
                actions=torch.cat([policy(obs[m*18:(m+1)*18]) for m,policy in enumerate(policies)])
                keep('requested_actions',actions)
                obs,_,reward,done,_,_,_=env.step(actions)
                for i in done.nonzero().flatten().tolist():resets.append(dict(index=i,step=step))
                values=dict(quat=env.base_quat,q=env.dof_pos,target=env.joint_pos_target,qd=env.dof_vel,
                    torque=env.torques,fz=env.contact_forces[:,env.feet_indices,2],gap=env.support_foot_gap,
                    omega=env.base_ang_vel,velocity=env.base_lin_vel,height=env.root_states[:,2],reset=done)
                for k,v in values.items():
                    assert torch.isfinite(v).all(),(step,k)
                    keep(k,v)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                if (step+1)%100==0:print('STEP',step+1,flush=True)
        assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        np.savez_compressed(out/'traces.npz',**{k:np.asarray(v) for k,v in records.items()})
        (out/'status.json').write_text(json.dumps(dict(completed=True,finite=True,resets=resets,training_updates=0,checkpoints_unchanged=True),indent=2)+'\n')
        print('COMPLETE',out,flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
