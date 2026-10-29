"""Review checks and frozen-policy rollout only; never call PPO update or save."""
import json,hashlib,sys,math
from pathlib import Path
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.envs.lz_q1l.lz_q1l_2 import LZQ1LSoftSlipRobot
from legged_gym.utils.helpers import class_to_dict
from check_lz_q1l_front_pose import diff,same
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'media/soft_slip_review_20261029';OUT.mkdir(exist_ok=True)
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
cfg,train=task_registry.get_cfgs('lz_q1l_flat_soft_slip');old,_=task_registry.get_cfgs('lz_q1l_flat_front_pose');c=cfg.rewards
changes=diff(class_to_dict(old),class_to_dict(cfg));assert set(changes)=={'.rewards.soft_slip_full_force_n','.rewards.soft_slip_speed_tolerance','.rewards.soft_slip_speed_scale','.rewards.scales.lateral_soft_slip'}
assert c.scales.lateral_front_pose==-.03 and c.scales.base_height==-100 and c.scales.tracking_lateral==3
assert not any(k in class_to_dict(c.scales) for k in ['feet_slide','lateral_feet_slide'])
write('config_audit.json',dict(changes=changes,env=class_to_dict(cfg),train=class_to_dict(train)))
f=LZQ1LSoftSlipRobot.soft_slip_values
force=torch.tensor([[0.,1.,5.,10.]]);v=torch.tensor([[[.13,0.]]*4]);active=torch.ones(1,dtype=torch.bool)
cost,w,speed=f(force,v,active,c);assert torch.allclose(w,torch.tensor([[0.,.104,1.,1.]]),atol=1e-6);assert torch.allclose(cost,w,atol=1e-6)
assert (f(force,v,~active,c)[0]==0).all()
assert (f(force,torch.full_like(v,.01),active,c)[0]==0).all()
assert (f(-force,v,active,c)[0]==0).all()
assert torch.allclose(f(force,v.flip(-1),active,c)[0],cost)
perm=torch.tensor([1,0,3,2]);assert torch.allclose(f(force[:,perm],v[:,perm],active,c)[0],cost[:,perm])
# Linear tail stays increasing; no ceiling or derivative jump at normalization=1.
assert torch.allclose(f(force,v*0+torch.tensor([.23,0.]),active,c)[0],3*w,atol=1e-5)
for threshold in [0.,5.]:
 a=f(torch.full((1,4),threshold+1e-5),v,active,c)[0];b=f(torch.full((1,4),threshold-1e-5),v,active,c)[0];assert (a-b).abs().max()<1e-4
trace_path=ROOT/'media/front_pose300_gait_eval_20261029/run_20261029_181002_5AGz/result/trace.json';trace=json.loads(trace_path.read_text());cal=[]
for seg in range(2):
 rows=trace[seg*250+50:(seg+1)*250];forces=torch.tensor([r['contact_force_z'] for r in rows]);vel=torch.tensor([r['foot_bottom_velocity_world'] for r in rows])[:,:,:2]
 values,weights,speeds=f(forces,vel,torch.ones(200,dtype=torch.bool),c);weighted=values*c.scales.lateral_soft_slip
 bins={label:float((weighted*mask).sum(1).mean()) for label,mask in [('force_under_1N',forces<1),('force_1_to_10N',(forces>=1)&(forces<10)),('force_at_least_10N',forces>=10)]}
 cal.append(dict(direction=['left','right'][seg],weighted_rate_perfoot=weighted.mean(0).tolist(),weighted_rate_total=float(weighted.sum(1).mean()),worst_step_rate=float(weighted.sum(1).min()),per_control_step_mean=float(weighted.sum(1).mean()*.02),force_bins=bins))
write('trace_calibration.json',dict(rows=cal,feet=['LH','RH','LF','RF'],trace=str(trace_path),trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),units='weighted reward rate before dt=.02 unless explicitly per_control_step',scope='fixed trace calibration, not learning outcome'))
write('synthetic_checks.json',dict(passed=True,checks=['zero_air_force','inactive_zero','speed_tolerance','force_smoothness','linear_tail','foot_permutation','xy_rotation']))
print('CPU_TRACE_PASS',json.dumps(cal),flush=True)
from legged_gym.utils import get_args
from rsl_rl.runners.him_side_coordination_runner import HIMSideCoordinationRunner
source=ROOT/'HIMLoco/legged_gym'/train.runner.source_checkpoint;assert hashlib.sha256(source.read_bytes()).hexdigest()==train.runner.source_sha256
sys.argv=[sys.argv[0],'--task','lz_q1l_flat_soft_slip','--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0','--num_envs','32'];args=get_args();cfg.env.num_envs=32;env=None
try:
 env,_=task_registry.make_env('lz_q1l_flat_soft_slip',args=args,env_cfg=cfg);runner=HIMSideCoordinationRunner(env,class_to_dict(train),None,env.device);model=runner.alg.actor_critic;saved=torch.load(source,map_location=env.device)
 assert saved['iter']==300 and same(saved['model_state_dict'],model.state_dict()) and same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict()) and same(saved['estimator_optimizer_state_dict'],model.estimator.optimizer.state_dict())
 assert not hasattr(runner.alg,'teacher') and all(not p.requires_grad for p in model.estimator.parameters())
 with torch.inference_mode():
  obs,_=env.reset();policy=env.build_inference_policy(model)
  for _ in range(100):
   action=policy(obs);assert torch.equal(action,model.act_inference(obs));obs,_,reward,_,_,_,_=env.step(action)
   assert torch.isfinite(reward).all() and torch.isfinite(env.soft_slip_cost).all() and (env.soft_slip_cost>=0).all()
   assert int((env.commands[:,1]>0).sum())==16 and int((env.commands[:,1]<0).sum())==16
  before=env.soft_slip_cost.clone();env._reward_lateral_soft_slip();assert torch.equal(before,env.soft_slip_cost)
  env.reset_idx(torch.tensor([0],device=env.device));assert (env.soft_slip_cost[0]==0).all()
  env._resample_commands(torch.tensor([1],device=env.device));assert (env.soft_slip_cost[1]==0).all()
  env.commands[2,:3]=0;env.commands[3,:3]=torch.tensor([.3,0,0],device=env.device);env._post_physics_step_callback();assert (env.soft_slip_cost[2:4]==0).all()
 assert same(saved['model_state_dict'],model.state_dict()) and same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
 write('gpu_checks.json',dict(passed=True,envs=32,steps=100,ppo_updates=0,exact_restore=True,weights_unchanged=True,optimizer_unchanged=True,estimator_frozen=True,teacher_absent=True,native_actions=True,lifecycle_passed=True,metrics=env.pop_side_coordination_metrics()))
 files=set(json.loads(source.with_name('run_metadata.json').read_text())['code_sha256']);files.add('tools/check_lz_q1l_soft_slip.py')
 write('review_manifest.json',dict(source=str(source),source_sha256=train.runner.source_sha256,code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(files)}))
 print('FROZEN_GPU_PASS_NO_TRAINING',flush=True)
finally:
 if env is not None:env.gym.destroy_sim(env.sim)
