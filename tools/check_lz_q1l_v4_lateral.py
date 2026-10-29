"""CPU contract checks and optional 64-env simulation; no experiment training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror,LeftMirroredLZQ1LPolicy
from legged_gym.utils.lz_q1l_stand_walk import StandWalkResidualActor
from legged_gym.utils.lz_q1l_lateral import LateralResidualActor
from rsl_rl.modules.him_lateral import HIMLateralActorCritic
from rsl_rl.algorithms.him_lateral_ppo import HIMLateralPPO

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'media/v4_lateral_review_20261027/direct_no_mirror'
BASE=ROOT/'HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu-sim',action='store_true');opts=parser.parse_args()
    torch.manual_seed(21);torch.set_num_threads(2)
    cfg,train=task_registry.get_cfgs('lz_q1l_flat_v4_lateral')
    bc,_=task_registry.get_cfgs('lz_q1l_flat_stand_walk')
    from train_lz_q1l_v1_support_500 import differences
    diff=differences(class_to_dict(bc),class_to_dict(cfg))
    assert class_to_dict(bc.sim)==class_to_dict(cfg.sim)
    assert class_to_dict(bc.control)==class_to_dict(cfg.control)
    assert cfg.rewards.scales.base_height==bc.rewards.scales.base_height
    path=BASE/'model_800.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
    state=torch.load(path,map_location='cpu')['model_state_dict']
    names=[kind+'_leg_'+leg+'_joint' for leg in ('b_l','b_r','f_l','f_r') for kind in ('roll','pitch01','pitch02')]
    # Scale/mapping are checkpoint buffers; order matches its named-joint configuration.
    model=HIMLateralActorCritic(cfg.env.num_observations,cfg.env.num_privileged_obs,cfg.env.num_one_step_observations,12,**class_to_dict(train.policy))
    model.configure_lateral(names,state['command_scale']);model.load_new_baseline(state)
    assert torch.equal(model.right_rear_indices,torch.tensor([3,4,5]))
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(('critic.','lateral_delta.')) and k!='lateral_log_std'}
    mirror=LZQ1LMirror(names)
    def wrapped(adapter):return LeftMirroredLZQ1LPolicy(adapter,mirror,model.command_scale)
    old=wrapped(StandWalkResidualActor(model,.02));new=LateralResidualActor(model,.02)
    history=torch.randn(32,270)*.1
    with torch.no_grad():
        # Nonzero side residual ensures preservation is structural, not accidental zero init.
        model.lateral_delta[-1].weight.normal_(0,.01);model.lateral_delta[-1].bias.normal_(0,.01)
        for speed in [0.]*5+[.2]*5+[0.]*3+[.4]*18+[.5]*3+[0.]*18:
            history[:,:3]=0.;history[:,0]=speed*model.command_scale[0]
            assert torch.equal(old(history),new(history))
        old.actor.reset(torch.tensor([1,3]));new.reset(torch.tensor([1,3]))
        assert torch.equal(old(history),new(history))
        right=history.clone();right[:,:3]=0.;right[:,1]=-.2*model.command_scale[1]
        left=mirror.history(right)
        a=model.lateral_actions(right);b=model.lateral_actions(left)
        assert not torch.allclose(b,mirror.joints(a),atol=1e-6,rtol=0.), 'Unexpected hard mirror constraint'
        for h,actual in [(right,a),(left,b)]:
            v,z=model.estimator(h)
            direct=model.actor(torch.cat((h[:,:45],v,z),dim=-1))+model.lateral_delta(torch.cat((h,v,z),dim=-1))
            assert torch.equal(actual,direct), 'Observation/action direction was transformed'
        assert torch.allclose(new(right),a,atol=1e-6,rtol=0.)
        assert torch.allclose(new(left),b,atol=1e-6,rtol=0.)
        model.lateral_log_std.copy_(torch.linspace(-2.,-1.,12))
        model.update_distribution(right);std=model.action_std.clone()
        model.update_distribution(left);assert torch.equal(model.action_std,std), 'Joint noise was mirrored'
        assert not new.mirror_mask(left).any()
    algo_cfg=class_to_dict(train.algorithm);algo_cfg.update(num_learning_epochs=1,num_mini_batches=1)
    alg=HIMLateralPPO(model,**algo_cfg);alg.rollout_policy=new
    alg.init_storage(32,4,[270],[cfg.env.num_privileged_obs],[12])
    before_delta=model.lateral_delta[-1].weight.detach().clone()
    obs=history.clone();obs[:8,:3]=0.;obs[8:16,0]=.3*model.command_scale[0]
    obs[16:24]=right[16:24];obs[24:]=left[24:]
    crit=torch.randn(32,cfg.env.num_privileged_obs)*.1
    with torch.no_grad():
        for _ in range(4):
            actions=alg.act(obs,crit)
            # Same distribution for PPO recomputation on lateral samples.
            mean=alg.transition.action_mean.clone();mask=model.lateral_mask(obs)
            model.update_distribution(obs[mask]);assert torch.allclose(model.action_mean,mean[mask],atol=1e-6)
            assert torch.equal(actions[~mask],mean[~mask])
            alg.process_env_step(torch.randn(32),torch.zeros(32,dtype=torch.uint8),{},crit)
        alg.compute_returns(crit)
    alpha=new.alpha.clone();losses=alg.update()
    assert torch.equal(alpha,new.alpha), 'PPO update advanced live blend state'
    assert not torch.equal(before_delta,model.lateral_delta[-1].weight)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in frozen.items())
    assert all(p.grad is None for p in model.frozen_policy_parameters())
    checks=dict(new_baseline_exact_nonlateral=True,partial_reset=True,direct_observations_and_actions=True,
        no_mirror_constraint=True,no_noise_permutation=True,rollout_update_lateral_means_equal=True,deterministic_nonlateral=True,
        frozen_after_synthetic_optimizer_step=True,blend_not_advanced_by_ppo=True,config_diff=diff,
        synthetic_losses=list(losses),experiment_training_updates=0)
    # Synthetic optimizer check is discarded; never saved as a trained model.
    if opts.gpu_sim:
        from legged_gym.utils import get_args
        sys.argv=[sys.argv[0],'--task','lz_q1l_flat_v4_lateral','--num_envs','64','--headless','--sim_device','cuda:0','--rl_device','cuda:0']
        args=get_args();cfg.env.num_envs=64;cfg.noise.add_noise=False;cfg.commands.resampling_time=1e6;cfg.sim.physx.num_threads=2
        for key in dir(cfg.domain_rand):
            if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
        env=None
        try:
            env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
            runner,_=task_registry.make_alg_runner(env,args=args,train_cfg=train,log_root=None)
            fresh=runner.alg.actor_critic
            assert all(torch.equal(fresh.state_dict()[k].cpu(),v) for k,v in state.items())
            assert (fresh.lateral_delta[-1].weight==0).all()
            obs,_=env.reset();resets=0;support_samples=0;side_nonzero=False
            # Arbitrary asynchronous resets: total sampled side directions differ by at most one.
            left_count=right_count=0
            for n in [1,3,7,16,63,64]*10:
                ids=torch.randperm(64,device=env.device)[:n]
                env._resample_commands(ids)
                left_count+=int((env.command_sample_mode[ids]==2).sum())
                right_count+=int((env.command_sample_mode[ids]==3).sum())
                assert abs(left_count-right_count)<=1
            with torch.no_grad():
                for step in range(160):
                    env.commands.zero_()
                    env.commands[16:32,0]=.3;env.commands[32:48,1]=.2;env.commands[48:,1]=-.2
                    env.command_sample_mode[:16]=0;env.command_sample_mode[16:32]=1;env.command_sample_mode[32:48]=2;env.command_sample_mode[48:]=3
                    obs[:,:3]=env.commands[:,:3]*env.commands_scale
                    obs,_,reward,done,_,_,_=env.step(runner.get_inference_policy()(obs))
                    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                    assert (env.lateral_gait_penalties[:32]==0).all()
                    assert (env._reward_lateral_error()[:32]==0).all()
                    side_nonzero |= bool((env.lateral_gait_penalties[32:]>0).any())
                    resets+=int(done.sum());support_samples+=64
            assert side_nonzero
            assert all(torch.equal(fresh.state_dict()[k].cpu(),v) for k,v in state.items())
            checks['simulation']=dict(envs=64,steps=160,resets=resets,finite=True,lateral_reward_gate=True,untrained_lateral=True,
                sampled_left=left_count,sampled_right=right_count,balanced_direction_sampling=True)
        finally:
            if env is not None:env.gym.destroy_sim(env.sim)
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    print('V4_CHECKS',json.dumps(checks),flush=True)

if __name__=='__main__':main()
