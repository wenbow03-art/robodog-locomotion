"""Record continuous RH acceleration model500 at selected acceleration,20 trials,continuous zero-start video."""
import argparse
import json
import os
import random,secrets
import subprocess
import shutil
import sys
from pathlib import Path
import isaacgym
from isaacgym import gymtorch, gymapi
import torch
import numpy as np
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2_low_path import LowPathObservations, LZQ1LUniformPathRobot,TASK,register_task
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.modules.him_stance_width_actor import HIMStanceWidthActorCritic
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LForwardAccelerationRobot
from train_lz_q1l_three_band import SOURCE,SOURCE_SHA
from lz_q1l_acceleration_common import STANDING, STANDING_SHA
from lz_q1l_continuous_rh_common import attach_standing, digest, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    opts = p.parse_args()
    opts.out.mkdir(parents=True, exist_ok=False)
    register_task(task_registry)
    assert digest(SOURCE)==SOURCE_SHA
    final=opts.job/'final/model700_bands.pt'
    assert digest(final)==json.loads((final.parent/'manifest.json').read_text())['sha256']
    checkpoints=[SOURCE,final]
    cfg, train = task_registry.get_cfgs(TASK)
    c = cfg.acceleration
    c.standing_seconds = 5.0
    c.evaluation_seconds = 25.0
    trials = c.evaluation_trials
    n = trials*7*len(checkpoints)
    cfg.env.num_envs = n
    cfg.env.episode_length_s = c.evaluation_seconds+2
    cfg.noise.add_noise = False
    cfg.sim.physx.num_threads = 2
    for k in dir(cfg.domain_rand):
        if not k.startswith('_') and isinstance(getattr(cfg.domain_rand,k), bool):
            setattr(cfg.domain_rand,k,False)
    starts = np.random.RandomState(417).uniform(-.03,.03,(trials,6));starts[0] = 0.
    class EvaluationRobot(LZQ1LUniformPathRobot):
        def step(self,actions):
            # The exported policy includes its own stand/walk blend.
            return super(LZQ1LForwardAccelerationRobot,self).step(actions)
        def check_termination(self):
            super().check_termination()
            self.audit_term_force=self.contact_forces[:, self.termination_contact_indices, :].norm(dim=-1).clone()

        schedule_tick=0
        def _next_ramp_commands(self):
            self.commands.zero_()
            k=max(self.schedule_tick-249,0)
            tick=self.schedule_tick;stage=min(tick//250,4)
            cmd=self.commands[:,0].view(-1,7,trials)
            for i,target in enumerate((.5,.75,1.,1.25,1.5,2.)):cmd[:,i,:]=min(k*.02,target)
            profile=(0.,.4,.8,1.25,2.,1.25,.75,.4);stage=0 if tick<250 else min(1+(tick-250)//150,7)
            old=profile[max(0,stage-1)];tar=profile[stage]
            cmd[:,6,:]=old+np.sign(tar-old)*min(abs(tar-old),.02*(1+(tick-250)%150)) if tick>=250 else 0.
            self.command_sample_mode[:]=(self.commands[:,0]>0).long()
        def _reset_root_states(self, ids):
            self.root_states[ids] = self.base_init_state
            self.root_states[ids,:3] += self.env_origins[ids]
            self.root_states[ids,7:13] = torch.tensor(starts,dtype=torch.float,device=self.device)[ids%trials]
            ix=ids.to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_states),gymtorch.unwrap_tensor(ix),len(ix))
    task_registry.task_classes[TASK] = EvaluationRobot
    sys.argv = [sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
    args = get_args();env=None;encoder=None
    try:
        env,_ = task_registry.make_env(TASK,args=args,env_cfg=cfg)
        raw=env;base=HIMStanceWidthActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,12,**class_to_dict(train.policy)).to(env.device)
        base.configure_right_rear(env.dof_names,env.commands_scale);base.load_state_dict(torch.load(SOURCE,map_location=env.device)['model_state_dict']);base.eval()
        attach_standing(raw);pack=torch.jit.load(str(final),map_location=env.device).eval()
        half=7*trials
        hist=torch.ones(half,14,device=env.device,dtype=torch.long);alpha=torch.zeros(half,2,device=env.device);base_alpha=torch.zeros(half,1,device=env.device)
        frames=[];dones=[]
        with torch.inference_mode():
            obs,_=env.reset()
            for step in range(1250):
                raw.schedule_tick=step+1
                h1,h2=obs.split(half)
                base_alpha=(base_alpha+env.dt/.3).clamp(0.,1.)*(env.commands[:half,:1]>0)
                a1=(1-base_alpha)*raw.ramp_standing_policy(h1)+base_alpha*base.act_inference(h1)
                errors=raw.ramp_step_metrics[half:][:,[3,5,6]]*(env.commands[half:,:1]>0)
                a2,hist,alpha=pack(h2,errors,hist,alpha);actions=torch.cat((a1,a2))
                obs,_,_,done,_,_,_=env.step(actions)
                hist[done[half:].bool()]=1;alpha[done[half:].bool()]=0;base_alpha[done[:half].bool()]=0
                frames.append(env.ramp_step_metrics.cpu().numpy().copy())
                dones.append(done.cpu().numpy().copy())
                if (step+1)%100==0:print('STEP',step+1,flush=True)
        np.savez_compressed(opts.out/'trace.npz',metrics=frames,done=dones)
        write(opts.out/'setup.json',dict(checkpoints=[str(x) for x in checkpoints],command_accel=1.,protocols=[".5",".75","1","1.25","1.5","2","cross_both_directions"],starts=starts.tolist()))
        metrics=np.asarray(frames).reshape(1250,len(checkpoints),7,trials,8)
        done=np.asarray(dones).reshape(1250,len(checkpoints),7,trials).astype(bool)
        summary=[]
        for i,ck in enumerate(checkpoints):
            for j,(label,cut,end) in enumerate([('.5',275,1175),('.75',288,1188),('1',300,1200),('1.25',313,1213),('1.5',325,1225),('2',350,1250),('cross_both_directions',250,1250)]):
                m=metrics[:,i,j];ok=~done[:end,i,j].any(0)
                summary.append(dict(checkpoint=ck.name,protocol=label,no_reset=int(ok.sum()),actual_vx=float(m[cut:end,ok,1].mean()) if ok.any() else None,end_abs_y=float(abs(m[end-1,ok,6]).mean()) if ok.any() else None,peak_abs_y=float(abs(m[250:end,ok,6]).max(0).mean()) if ok.any() else None,end_yaw_deg=float(np.degrees(m[end-1,ok,5]).mean()) if ok.any() else None))
        write(opts.out/'summary.json',summary)
    finally:
        if encoder is not None and encoder.poll() is None:encoder.terminate();encoder.wait(timeout=15)
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
