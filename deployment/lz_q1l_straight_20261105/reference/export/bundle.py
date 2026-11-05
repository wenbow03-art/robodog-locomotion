"""Export the stand/800/700/two-band policy as one TorchScript file, with explicit controller state."""
import argparse, json, os
from pathlib import Path
import isaacgym
import torch
from torch import nn
from torch.nn.functional import normalize
from band700.routing import route,bounds
from train_lz_q1l_three_band import SOURCE,SOURCE_SHA
from lz_q1l_acceleration_common import STANDING,digest,write
from rsl_rl.modules.him_low_path import LowActor
from rsl_rl.modules.him_right_rear_only import HIMRightRearOnlyActorCritic
from legged_gym.envs import task_registry
from legged_gym.utils.lz_q1l_stand_walk import StandWalkResidualActor

class Combined(nn.Module):
    def __init__(self,old,models):
        super().__init__()
        new=models[0]
        self.e0=old.estimator.encoder;self.a0=old.actor;self.r0=old.right_rear_delta
        self.e=new.estimator.encoder;self.a=new.actor;self.r=new.right_rear_delta
        self.register_buffer('scale',new.command_scale)
        self.other=nn.ModuleList([m.low_other for m in models]);self.rh=nn.ModuleList([m.low_rh for m in models])
    def features(self,h,p):
        return torch.cat((h[:,:45],p[:,:3],normalize(p[:,3:],dim=-1)),dim=1)
    def forward(self,h,errors,hist,alphas):
        cmd=h[:,:3]/self.scale;vx=cmd[:,0];straight=(cmd[:,1:].abs()<=.05).all(1)
        standing=(cmd.abs()<=.05).all(1);moving=vx>0
        weights,hist=route(vx,hist);w=weights[:,0:1]
        fw=(vx>=.1)&straight
        oldalpha=(alphas[:,0]+(fw.float()-alphas[:,0]).clamp(-1/15.,1/15.)).clamp(0.,1.)
        oldalpha=torch.where(oldalpha>=1.-1e-6,torch.ones_like(oldalpha),oldalpha)
        oldalpha=torch.where((oldalpha<=1e-6)|~(standing|fw),torch.zeros_like(oldalpha),oldalpha)
        a=((alphas[:,1]+1/15.).clamp(0.,1.)*moving.float())[:,None]
        fo=self.features(h,self.e0(h));stand=self.a0(fo)
        low=stand.clone();low[:,3:6]+=oldalpha[:,None]*self.r0(fo)
        f=self.features(h,self.e(h));base=self.a(f)
        base[:,3:6]+=self.r(f)*(standing|(straight&(vx>.05)))[:,None]
        out=w*low+(1.-w)*((1.-a)*stand+a*base)
        x=torch.cat((f,errors.clamp(-5.,5.)),1)
        for i,(rh,other) in enumerate(zip(self.rh,self.other)):
            o=other(x);delta=torch.cat((o[:,:3],rh(x),o[:,3:]),1).tanh()*.4
            out=out+a*weights[:,i+1:i+2]*delta
        return out,hist,torch.cat((oldalpha[:,None],a),1)

def export(job,smoke=False):
    torch.manual_seed(1);torch.set_num_threads(2)
    assert digest(SOURCE)==SOURCE_SHA
    opt=json.loads((SOURCE.parent/'train_config.json').read_text())['policy']
    names=[k+'_leg_'+l+'_joint' for l in ('b_l','b_r','f_l','f_r') for k in ('roll','pitch01','pitch02')]
    def load(p):return torch.load(p,map_location='cpu')['model_state_dict']
    base=load(SOURCE);oldck=STANDING.parent/'model_800.pt'
    assert digest(oldck)=='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
    old=HIMRightRearOnlyActorCritic(270,238,45,12,**opt);old.configure_right_rear(names,base['command_scale']);old.load_state_dict(load(oldck))
    stand=load(STANDING)
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in stand.items() if k.startswith(('actor.','estimator.')))
    models=[];checks={}
    for idx in (1,2):
        model=LowActor(274,238,45,12,**opt);model.configure_right_rear(names,base['command_scale'])
        if smoke:
            model.load_state_dict(base,strict=False)
            with torch.no_grad():
                model.low_rh[-1].bias.fill_(.23*idx);model.low_other[-1].bias.fill_(-.15*idx)
        else:
            folder=job/f'band{idx}';status=json.loads((folder/'status.json').read_text())
            assert status['state']=='completed'
            completed=int(status['completed_updates']);ck=folder/f'model_{completed}.pt'
            sd=torch.load(ck,map_location='cpu')
            assert sd['iter']==completed and sd['infos']['source_sha256']==SOURCE_SHA and sd['infos']['band']==idx and sd['infos']['bounds']==list(bounds(idx))
            assert all(torch.equal(sd['model_state_dict'][k],v) for k,v in base.items() if not k.startswith('critic.'))
            model.load_state_dict(sd['model_state_dict']);checks[str(idx)]={'checkpoint':ck.name,'sha256':digest(ck)}
        assert model.right_rear_indices.tolist()==[3,4,5];models.append(model.eval())
    policy=Combined(old.eval(),models).eval();out=job/('export_test' if smoke else 'final');out.mkdir(exist_ok=True)
    h=torch.randn(8,270);err=torch.randn(8,3);hist=torch.ones(8,14,dtype=torch.long);alpha=torch.zeros(8,2)
    with torch.no_grad():
        graph=torch.jit.script(policy)
        temp=out/'policy.tmp';torch.jit.save(graph,str(temp));deploy=torch.jit.load(str(temp))
        oldroute=StandWalkResidualActor(old,.02)
        worst=0.
        for v,n in [(0.,20),(.25,20),(.5,20),(.52,20),(.75,20),(1.,20),(1.4,20),(1.5,20),(2.,20),(.4,2),(1.2,2),(.7,5),(0.,1)]:
            for _ in range(n):
                h[:,:3]=0;h[:,0]=v*base['command_scale'][0]
                w,hn=route(h[:,0]/base['command_scale'][0],hist);lo=oldroute.act_inference(h)
                a=((alpha[:,1]+1/15.).clamp(0.,1.)*(v>0))[:,None]
                zero=h.new_zeros(len(h),1);baseline=models[0].act_inference(torch.cat((h,err,zero),1))
                feat=old.estimator(h);st=old.actor(torch.cat((h[:,:45],*feat),1))
                expected=w[:,:1]*lo+(1-w[:,:1])*((1-a)*st+a*baseline)
                for i,m in enumerate(models):
                    expected+=a*(m.act_inference(torch.cat((h,err,w[:,i+1:i+2]),1))-baseline)
                actual,hist,alpha=deploy(h,err,hist,alpha)
                worst=max(worst,float((actual-expected).abs().max()));assert torch.equal(hist,hn)
        assert worst<4e-6,worst
    dest=out/'model700_bands.pt';os.replace(temp,dest)
    write(out/'manifest.json',dict(test_only=smoke,source700=SOURCE_SHA,baseline800=digest(oldck),standing=digest(STANDING),experts=checks,sha256=digest(dest),equivalence_max_error=worst,interface='At 50 Hz: (history[N,270], signed yaw_rate rad/s, heading rad and offset m as errors[N,3], history_ids[N,14] int64 initialized to ones, alphas[N,2] initialized to zeros) -> (executed action[N,12], next_ids, next_alphas); reset state per episode. Straight commands only. Do not apply an additional stand mixer.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--smoke',action='store_true');args=p.parse_args();export(args.job,args.smoke)
