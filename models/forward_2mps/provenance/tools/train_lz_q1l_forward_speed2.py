"""Authorized300 updates from speed1m/s300 to .2..2.0; restore moving teacher exactly."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from train_lz_q1l_stance_width import write,same

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_forward_speed/forward_speed_300_20261030_212312_3965219/model_300.pt'
SHA='0e77f5d8748fb5d43fda25efcb108a0d34a8272abdbe22c765d8f58c859c75c3'
TASK='lz_q1l_flat_forward_speed2'
FILES=['tools/train_lz_q1l_forward_speed2.py','tools/run_lz_q1l_forward_speed2.sh',
    'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ema_rear_teacher_ppo.py',
    'HIMLoco/rsl_rl/rsl_rl/runners/him_ema_rear_teacher_runner.py']

def rules(runner,env):
    import torch
    alg=runner.alg;model=alg.actor_critic
    obs=torch.randn(32,env.num_obs,device=env.device)*.05
    obs[:,:3]=0.;obs[:,0]=1.8*env.commands_scale[0]
    teacher_count_before=alg.teacher_updates
    assert not any(p.requires_grad for p in alg.teacher.parameters())
    assert {p.data_ptr() for p in model.parameters()}.isdisjoint({p.data_ptr() for p in alg.teacher.parameters()})
    # Reflection is an involution and the RH target comes specifically from teacher LH.
    assert torch.equal(alg.mirror.history(alg.mirror.history(obs)),obs)
    target,_=alg.teacher_targets(obs)
    with torch.no_grad():
        mirrored=alg.teacher.act_inference(alg.mirror.history(obs))
        lh=[env.dof_names.index(k+'_leg_b_l_joint') for k in ('roll','pitch01','pitch02')]
        expected=mirrored[:,lh]*mirrored.new_tensor([-1.,1.,1.])
        assert torch.equal(target,expected)
    model.zero_grad()
    loss,stats=alg.distillation_loss(obs,model.act_inference(obs));loss.backward()
    assert abs(stats['weighted_loss']-.3*stats['rear_mse'])<1e-5
    assert all(p.grad is None or not p.grad.any() for p in model.actor.parameters())
    assert all(p.grad is None for p in alg.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.right_rear_delta.parameters())
    model.zero_grad()
    # Ordinary policy loss still reaches LH; no left-leg freeze is introduced.
    model.act_inference(obs)[:,lh].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.actor.parameters())
    assert all(p.grad is None for p in model.estimator.parameters())
    model.zero_grad()
    # Teacher follows exactly once per update using EMA, with no source mutation.
    teacher_before={k:v.clone() for k,v in alg.teacher.state_dict().items()}
    source_param=next(model.actor.parameters());original=source_param.detach().clone()
    old_teacher_param=next(alg.teacher.actor.parameters()).detach().clone()
    with torch.no_grad():source_param.add_(.01)
    alg.update_teacher()
    teacher_param=next(alg.teacher.actor.parameters())
    assert torch.allclose(teacher_param,old_teacher_param*(1-alg.teacher_ema_tau)+(original+.01)*alg.teacher_ema_tau,atol=2e-7)
    with torch.no_grad():source_param.copy_(original)
    alg.teacher.load_state_dict(teacher_before);alg.teacher_updates=teacher_count_before
    assert same(teacher_before,alg.teacher.state_dict())
    return dict(mirror_target_is_LH=True,teacher_independent=True,imitation_only_RH_branch=True,
        LH_trainable=True,teacher_stop_gradient=True,forward_weight=.3,ema_exact=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--preflight',action='store_true')
    opt=p.parse_args();out=opt.run_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    target=2 if opt.preflight else 300
    write(out/'status.json',dict(state='initializing',completed_updates=0,target_updates=target,preflight=opt.preflight))
    env=None
    try:
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
        import isaacgym
        import torch
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        from legged_gym.utils.helpers import class_to_dict
        from rsl_rl.runners.him_ema_rear_teacher_runner import HIMEmaRearTeacherRunner
        from train_lz_q1l_v1_support_500 import differences
        cfg,train=task_registry.get_cfgs(TASK)
        old=json.loads((SOURCE.parent/'env_config.json').read_text());old['domain_rand'].pop('push_interval',None)
        changes=differences(old,class_to_dict(cfg))
        assert set(changes)=={'commands.pure_forward_speed_range','commands.ranges.lin_vel_x'},changes
        assert cfg.commands.pure_forward_speed_range==[.2,2.] and cfg.commands.zero_command_prob==cfg.commands.pure_forward_prob==.5
        assert cfg.sim.dt==.001 and cfg.control.decimation==20 and cfg.env.num_envs==4096
        assert train.algorithm.rear_teacher_forward_coef==.3 and train.algorithm.rear_teacher_coef==.1
        assert train.runner.num_steps_per_env==100 and train.runner.save_interval==50
        old_train=json.loads((SOURCE.parent/'train_config.json').read_text())
        train_changes=differences(old_train,class_to_dict(train))
        assert set(train_changes)=={'runner.experiment_name'},train_changes
        if opt.preflight:cfg.env.num_envs=256
        previous=json.loads((SOURCE.parent/'run_metadata.json').read_text());files=set(FILES)|set(previous['code_sha256'])
        baseline=ROOT/'HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027/gait_150.pt'
        baseline_sha=hashlib.sha256(baseline.read_bytes()).hexdigest()
        assert baseline_sha=='655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747'
        metadata=dict(task=TASK,source_checkpoint=str(SOURCE),source_sha256=SHA,source_iteration=300,
            additional_iterations=target,preflight=opt.preflight,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            seed=1,configuration_changes=changes,train_configuration_changes=train_changes,policy_training='all_four_legs',estimator_frozen=True,std_frozen=True,
            teacher='EMA continued from source checkpoint, mirrored LH targets for RH correction only',teacher_ema_tau=.05,teacher_restored=True,
            teacher_forward_coefficient=.3,teacher_stand_coefficient=.1,initial_learning_rate=1e-5,
            save_interval=50,standing_baseline=str(baseline),standing_baseline_sha256=baseline_sha,
            code_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)})
        for f in files:
            dest=out/'provenance'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
        write(out/'run_metadata.json',metadata);write(out/'env_config.json',class_to_dict(cfg));write(out/'train_config.json',class_to_dict(train))
        sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
        args=get_args();env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMEmaRearTeacherRunner(env,class_to_dict(train),str(out),device=env.device)
        runner.load(str(SOURCE));model=runner.alg.actor_critic
        saved=torch.load(SOURCE,map_location=env.device);assert saved['iter']==300
        assert same(saved['model_state_dict'],model.state_dict())
        assert same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
        assert same(saved['estimator_optimizer_state_dict'],model.estimator.optimizer.state_dict())
        assert same(saved['teacher_state_dict'],runner.alg.teacher.state_dict())
        teacher_start=saved['teacher_updates'];assert teacher_start==runner.alg.teacher_updates==300
        saved_lr=runner.alg.learning_rate;runner.alg.learning_rate=1e-5
        for g in runner.alg.optimizer.param_groups:g['lr']=1e-5
        frozen={k:v.clone() for k,v in model.estimator.state_dict().items()}
        est_opt=copy.deepcopy(model.estimator.optimizer.state_dict());initial_actor={k:v.clone() for k,v in model.actor.state_dict().items()}
        checks=rules(runner,env)
        assert same(saved['model_state_dict'],model.state_dict())
        assert same(saved['teacher_state_dict'],runner.alg.teacher.state_dict()) and runner.alg.teacher_updates==teacher_start
        write(out/'resume_audit.json',dict(exact_model=True,original_adam_moments_restored=True,
            exact_estimator_optimizer=True,source_lr=saved_lr,initial_lr=1e-5,teacher_exactly_restored=True,teacher_source_updates=teacher_start,checks=checks))
        del saved
        runner.current_learning_iteration=0
        original_save,original_log=runner.save,runner.log
        original_save(str(out/'model_0.pt'),infos=metadata)
        sums=torch.zeros(5,6,device=env.device);counts=torch.zeros(5,device=env.device)
        original_step=env.step
        def step(actions):
            command=env.commands[:,0].clone();result=original_step(actions);valid=~result[3].bool()
            x,y,z,w=env.base_quat.unbind(-1)
            roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)).abs()*180/torch.pi
            packet=torch.stack((command,env.base_lin_vel[:,0],(env.base_lin_vel[:,0]-command).abs(),
                env.root_states[:,2],roll,env.base_lin_vel[:,1].abs()),dim=1)
            for i,mask in enumerate((command<=.05,(command>.05)&(command<=.5),(command>.5)&(command<=1.),(command>1.)&(command<=1.5),command>1.5)):
                mask=mask&valid;sums[i].add_((packet*mask[:,None]).sum(0));counts[i].add_(mask.sum())
            return result
        env.step=step
        def log(locs):
            original_log(locs);completed=locs['it']+1
            assert runner.alg.teacher_updates==teacher_start+completed
            assert same(frozen,model.estimator.state_dict()) and same(est_opt,model.estimator.optimizer.state_dict())
            assert not any(p.grad is not None for p in runner.alg.teacher.parameters())
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            speeds={}
            for i,name in enumerate(['stand','forward_020_050','forward_050_100','forward_100_150','forward_150_200']):
                if counts[i]>0:
                    row=(sums[i]/counts[i]).tolist()
                    speeds[name]=dict(zip(['command_vx','actual_vx','speed_mae','height_m','roll_abs_deg','lateral_abs_speed'],row))
                    for key,value in speeds[name].items():runner.writer.add_scalar('Speed/'+name+'/'+key,value,completed)
            sums.zero_();counts.zero_()
            width=env.pop_stance_width_metrics()
            for k,v in width.items():runner.writer.add_scalar('StanceWidth/'+k,v,completed)
            for k,v in runner.alg.distillation_metrics.items():runner.writer.add_scalar('Teacher/'+k,v,completed)
            if completed%50==0 or (opt.preflight and completed==target):
                old=runner.current_learning_iteration;runner.current_learning_iteration=completed
                try:
                    file=out/('model_%d.pt'%completed);assert not file.exists()
                    original_save(str(file)+'.tmp',infos=metadata);os.replace(str(file)+'.tmp',file)
                finally:runner.current_learning_iteration=old
                assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_sha
            import statistics
            write(out/'status.json',dict(state='running',completed_updates=completed,target_updates=target,preflight=opt.preflight,
                pid=os.getpid(),iteration_seconds=locs['collection_time']+locs['learn_time'],learning_rate=runner.alg.learning_rate,
                speed=speeds,width=width,teacher=runner.alg.distillation_metrics,
                mean_reward=statistics.mean(locs['rewbuffer']) if locs['rewbuffer'] else None,
                mean_episode_length=statistics.mean(locs['lenbuffer']) if locs['lenbuffer'] else None))
        runner.log=log;runner.save=lambda *a,**kw:None
        runner.learn(target,init_at_random_ep_len=True)
        assert any(not torch.equal(v,model.actor.state_dict()[k]) for k,v in initial_actor.items())
        assert runner.current_learning_iteration==target and runner.alg.teacher_updates==teacher_start+target
        if opt.preflight:
            state=torch.load(out/('model_%d.pt'%target),map_location=env.device)
            assert same(state['teacher_state_dict'],runner.alg.teacher.state_dict())
            runner.load(str(out/('model_%d.pt'%target)))
            assert same(state['model_state_dict'],model.state_dict()) and same(state['optimizer_state_dict'],runner.alg.optimizer.state_dict())
            assert same(state['teacher_state_dict'],runner.alg.teacher.state_dict()) and runner.alg.teacher_updates==teacher_start+target
            checks.update(passed=True,discarded_updates=target,actor_updated=True,estimator_unchanged=True,teacher_checkpoint_roundtrip=True)
            write(out/'checks.json',checks)
        else:
            for i in range(0,301,50):assert (out/('model_%d.pt'%i)).is_file()
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
        runner.writer.flush();runner.writer.close()
        status=json.loads((out/'status.json').read_text());status['state']='completed';write(out/'status.json',status)
        print('PREFLIGHT_PASSED' if opt.preflight else 'TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        status=json.loads((out/'status.json').read_text());status.update(state='failed',error=repr(exc));write(out/'status.json',status);raise
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
