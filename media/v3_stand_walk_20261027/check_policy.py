from pathlib import Path
import json,hashlib,sys
import isaacgym
import torch
import legged_gym.envs
from rsl_rl.modules.him_right_rear_only import HIMRightRearOnlyActorCritic
from legged_gym.utils.lz_q1l_stand_walk import StandWalkResidualActor
from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror,LeftMirroredLZQ1LPolicy
ROOT=Path('/home/xiaohengli/Program/robo_competition');OUT=Path(__file__).resolve().parent
p=json.loads((ROOT/'media/v3_right_rear_split_20261027/continuation_800.json').read_text());r=Path(p['run_dir']);ck=r/'model_800.pt';saved=torch.load(ck,map_location='cpu');m=json.loads((r/'run_metadata.json').read_text());base=torch.load(m['teacher_checkpoint'],map_location='cpu')['model_state_dict']
assert hashlib.sha256(ck.read_bytes()).hexdigest()=='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
frozen=[k for k in base if k=='std' or k.startswith(('actor.','estimator.'))];assert all(torch.equal(base[k],saved['model_state_dict'][k]) for k in frozen)
actor=HIMRightRearOnlyActorCritic(270,238,45,12,**json.loads((r/'train_config.json').read_text())['policy'])
names=json.loads((ROOT/'media/v3_split800_analysis_20261027/eval/manifest.json').read_text())['dof_names'];actor.configure_right_rear(names,saved['model_state_dict']['command_scale']);actor.load_state_dict(saved['model_state_dict'],strict=True);actor.eval()
h=StandWalkResidualActor(actor,.02,.3);torch.manual_seed(1);obs=torch.randn(4,270)*.1;obs[:,:3]=0
other=torch.tensor([0,1,2,6,7,8,9,10,11])
def baseline(obs):
 v,z=actor.estimator(obs);return actor.actor(torch.cat((obs[:,:45],v,z),-1))
with torch.no_grad():
 assert torch.equal(h.act_inference(obs),baseline(obs))
 obs[:,0]=.35*actor.command_scale[0]
 alphas=[]
 for i in range(15):
  out=h.act_inference(obs);alphas.append(float(h.alpha[0]));assert torch.equal(out[:,other],baseline(obs)[:,other])
 assert torch.equal(h.alpha,torch.ones(4)) and torch.equal(out,actor.act_inference(obs))
 obs[:,:3]=0
 for i in range(15):out=h.act_inference(obs);assert torch.equal(out[:,other],baseline(obs)[:,other])
 assert not h.alpha.any() and torch.equal(out,baseline(obs))
 obs[:,0]=.35*actor.command_scale[0]
 for i in range(15):h.act_inference(obs)
 h.reset(torch.tensor([1,3]));assert torch.equal(h.alpha,torch.tensor([1.,0.,1.,0.]))
 h.reset();assert not h.alpha.any()
 obs[:,0]=.35*actor.command_scale[0]
 for i in range(6):h.act_inference(obs)
 previous=h.alpha.clone();obs[:,:3]=0;h.act_inference(obs);assert torch.allclose(previous-h.alpha,torch.full((4,),1/15))
 obs[:,:3]=0;obs[:,1]=.2*actor.command_scale[1]
 mirror=LZQ1LMirror(names);policy=LeftMirroredLZQ1LPolicy(h,mirror,actor.command_scale)
 assert torch.equal(policy(obs),mirror.joints(baseline(mirror.history(obs))))
 assert not h.alpha.any()
 assert all(torch.equal(v,saved['model_state_dict'][k]) for k,v in actor.state_dict().items())
report=dict(checkpoint=str(ck),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),training_updates=0,blend_steps=15,blend_time_s=.3,frozen_tensors=len(frozen),checks=dict(standing_exact_baseline=True,walking_exact_model800=True,other_nine_outputs_unchanged=True,partial_reset=True,mid_transition_reversal=True,left_mirror_preserved=True,weights_unchanged=True),ramp=alphas)
(OUT/'policy_checks.json').write_text(json.dumps(report,indent=2)+'\n');print('HYBRID_CHECKS_OK',json.dumps(report))
