"""Fixed CPU posture evaluation and rule checks; no learning updates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LV2UnifiedRobot
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from train_lz_q1l_v1_support_500 import differences

ROOT=Path(__file__).resolve().parents[1]
TASK='lz_q1l_flat_v2_posture'
STARTS=np.stack([np.zeros(6),np.random.RandomState(1).uniform(-.03,.03,6),np.random.RandomState(2).uniform(-.03,.03,6)])
COMMANDS=[('stand',0.),('forward_035',.35),('forward_050',.5)]


class Probe(LZQ1LV2UnifiedRobot):
    def _resample_commands(self,ids):
        self.commands[ids]=0.
        self.command_sample_mode[ids]=0
        self._clear_support(ids)
        self.v2_air_steps[ids]=0

    def _reset_root_states(self,ids):
        self.root_states[ids]=self.base_init_state
        self.root_states[ids,:3]+=self.env_origins[ids]
        self.root_states[ids,7:13]=torch.as_tensor(STARTS,dtype=torch.float,device=self.device)[ids%3]
        index=ids.to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(index),len(index))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--check-rules',action='store_true')
    opts=parser.parse_args()
    out=opts.out.resolve();out.mkdir(parents=True,exist_ok=False)
    digest=hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest()
    cfg,train=task_registry.get_cfgs(TASK)
    old,old_train=task_registry.get_cfgs('lz_q1l_flat_v2_unified')
    diff=differences(class_to_dict(old),class_to_dict(cfg))
    assert set(diff)=={'commands.zero_command_prob','commands.pure_forward_prob','commands.pure_lateral_prob','commands.pure_yaw_prob',
        'rewards.soft_dof_pos_limit','rewards.scales.base_height','rewards.scales.dof_pos_limits'},diff
    assert cfg.rewards.scales.base_height==-10. and cfg.rewards.scales.dof_pos_limits==-2.
    assert cfg.rewards.soft_dof_pos_limit==.95
    assert cfg.commands.zero_command_prob==cfg.commands.pure_forward_prob==.5
    assert cfg.commands.pure_lateral_prob==cfg.commands.pure_yaw_prob==0.
    assert train.runner.max_iterations==500 and train.runner.save_interval==10
    assert cfg.sim.dt==.001 and cfg.control.decimation==20
    cfg.env.num_envs=9;cfg.env.episode_length_s=20.
    cfg.commands.resampling_time=1e6;cfg.noise.add_noise=False
    cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=False;cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    torch.set_num_threads(2)
    sys.argv=[sys.argv[0],'--task',TASK,'--seed','1','--headless','--sim_device','cpu','--rl_device','cpu','--pipeline','cpu','--num_threads','2']
    args=get_args();task_registry.task_classes[TASK]=Probe
    env=None
    try:
        env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        urdf=Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT/'HIMLoco/legged_gym')))
        xml=ET.parse(urdf).getroot()
        limits=torch.tensor([[float(xml.find("joint[@name='%s']/limit"%name).get(k)) for k in ('lower','upper')] for name in env.dof_names])
        mid=limits.mean(1);half=(limits[:,1]-limits[:,0])*.5
        expected=torch.stack((mid-.95*half,mid+.95*half),dim=1)
        assert torch.allclose(env.dof_pos_limits,expected,atol=1e-6)
        checks={}
        if opts.check_rules:
            ids=torch.arange(9);counts=torch.zeros(6,dtype=torch.long)
            for _ in range(1000):
                LZQ1LV2UnifiedRobot._resample_commands(env,ids)
                counts+=torch.bincount(env.command_sample_mode,minlength=6)
                assert not env.commands[:,1:].any()
                f=env.commands[env.command_sample_mode==1,0]
                assert ((f>=.2)&(f<=.5)).all()
            assert not counts[2:].any() and abs(float(counts[0])/counts.sum().item()-.5)<.02
            env.dof_pos[:]=mid
            assert not env._reward_dof_pos_limits().any()
            env.dof_pos[:,3]=limits[3,0]
            assert (env._reward_dof_pos_limits()>0).all()
            assert abs(env.reward_scales['base_height']-(-10*env.dt))<1e-8
            assert abs(env.reward_scales['dof_pos_limits']-(-2*env.dt))<1e-8
            checks=dict(command_draws=int(counts.sum()),counts=counts.tolist(),soft_limits='95% centered URDF range; zero at center, positive penalty near hard limit',scales='passed')
        runner=HIMOnPolicyRunner(env,class_to_dict(train),log_dir=None,device='cpu')
        saved=torch.load(opts.checkpoint,map_location='cpu')
        runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        policy=runner.get_inference_policy(device=env.device)
        obs,_=env.reset();alive=torch.ones(9,dtype=torch.bool);resets=[]
        data={k:[] for k in ('valid','height','q','contact','vx','vy','yaw','gap')}
        with torch.no_grad():
            for step in range(500):
                env.commands.zero_();env.command_sample_mode[:]=0
                if step>=100:
                    for i,(_,speed) in enumerate(COMMANDS):
                        env.commands[i*3:(i+1)*3,0]=speed
                        env.command_sample_mode[i*3:(i+1)*3]=int(speed>0)
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                assert not policy.mirror_mask(obs).any()
                obs,priv,reward,done,_,_,_=env.step(policy(obs))
                for i in (alive&done.bool()).nonzero().flatten().tolist():resets.append(dict(index=i,time_s=(step+1)*.02))
                alive &= ~done.bool()
                values=dict(valid=alive,height=env.root_states[:,2],q=env.dof_pos,contact=env.contact_forces[:,env.feet_indices,2]>1.,
                    vx=env.base_lin_vel[:,0],vy=env.base_lin_vel[:,1],yaw=env.base_ang_vel[:,2],gap=env.support_foot_gap)
                assert all(torch.isfinite(v).all() for v in (obs,priv,reward,env.dof_pos))
                for key,value in values.items():data[key].append(value.cpu().numpy().copy())
        arrays={k:np.asarray(v) for k,v in data.items()}
        np.savez_compressed(out/'traces.npz',**arrays)
        cases={}
        lower,upper=limits[:,0].numpy(),limits[:,1].numpy()
        for i,(name,speed) in enumerate(COMMANDS):
            ids=np.arange(i*3,(i+1)*3);valid=arrays['valid'][200:,ids]
            row=dict(trials=3,failures=sum(r['index'] in ids for r in resets),valid_fraction=float(valid.mean()),target_vx=speed)
            if valid.any():
                h=arrays['height'][200:,ids][valid];q=arrays['q'][200:,ids][valid]
                row.update(height_mean_m=float(h.mean()),height_p05_m=float(np.percentile(h,5)),
                    vx=float(arrays['vx'][200:,ids][valid].mean()),vy=float(arrays['vy'][200:,ids][valid].mean()),
                    yaw=float(arrays['yaw'][200:,ids][valid].mean()),
                    contact_rate=arrays['contact'][200:,ids][valid].mean(0).tolist(),
                    mean_gap_mm=(arrays['gap'][200:,ids][valid].mean(0)*1000).tolist(),
                    hard_limit_rate_per_joint=((q<=lower+.01)|(q>=upper-.01)).mean(0).tolist(),
                    joint_mean_deg=(q.mean(0)*180/np.pi).tolist())
            cases[name]=row
        assert hashlib.sha256(opts.checkpoint.read_bytes()).hexdigest()==digest
        report=dict(checkpoint=str(opts.checkpoint.resolve()),checkpoint_sha256=digest,iteration=saved['iter'],cases=cases,
            rule_checks=checks,config_diff_from_six_command_v2=diff,backend='CPU PhysX',physics_dt=.001,control_dt=.02,
            dof_names=env.dof_names,starts=STARTS.tolist(),resets=resets,
            protocol='Three paired initial root velocities per command; policy stands 2s then command for 8s; report 4-10s before first reset. No noise/domain randomization, no mirrored commands. CPU monitor, not a GPU performance equivalence claim.')
        sources=[Path(__file__).resolve(),ROOT/'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
            ROOT/'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
            ROOT/'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',urdf]
        report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(cases=cases,rule_checks=checks)),flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
