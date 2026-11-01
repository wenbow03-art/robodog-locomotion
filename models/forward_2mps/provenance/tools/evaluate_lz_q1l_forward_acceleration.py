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
from lz_q1l_acceleration_common import TASK, attach_standing, digest, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--stage', type=int, required=True, choices=[0,1,2])
    p.add_argument('--out', type=Path, required=True)
    opts = p.parse_args()
    opts.out.mkdir(parents=True, exist_ok=False)
    checksum = digest(opts.checkpoint)
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
        frames=[];dones=[]
        with torch.inference_mode():
            obs,_=env.reset()
            for _ in range(round(c.evaluation_seconds/env.dt)):
                obs,_,_,done,_,_,_=env.step(policy(obs))
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
        write(opts.out/'report.json',report)
        print('EVALUATION_COMPLETE',[(r['acceleration'],r['passed']) for r in groups],flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
