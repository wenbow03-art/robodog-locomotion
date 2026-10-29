"""Resume RH-only300 with teacher weights stand=.1, forward=.3; no other changes."""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
from train_lz_q1l_v1_support_500 import differences, write_json
from train_lz_q1l_v2_unified import ROOT, install_logging
from train_lz_q1l_v3_right_rear_only import SOURCE, SOURCE_SHA256, attach_teacher

TASK='lz_q1l_flat_v3_right_rear_split'
RESUME=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v3_right_rear_only/rh_only_from_gait150_300_20261027_110207_1719864/model_300.pt'
RESUME_SHA256='8d41715f4f7836aafd15356ee0593ac68d879268b9384774a69c70084b68d97b'
NEW_SOURCES=['tools/train_lz_q1l_v3_right_rear_split.py','tools/launch_lz_q1l_v3_right_rear_split_600.py','tools/start_lz_q1l_v3_right_rear_split_600.sh','HIMLoco/rsl_rl/rsl_rl/algorithms/him_right_rear_split_ppo.py']

def preflight():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
    assert hashlib.sha256(RESUME.read_bytes()).hexdigest()==RESUME_SHA256
    parent=json.loads((RESUME.parent/'run_metadata.json').read_text())
    for name,digest in parent['code_sha256'].items():
        current=ROOT/name; archived=RESUME.parent/'provenance'/name
        assert hashlib.sha256(archived.read_bytes()).hexdigest()==digest,name
        if name.endswith(('/envs/__init__.py','/lz_q1l_2_config.py')):
            assert current.read_bytes().startswith(archived.read_bytes()),name
        elif name.endswith('/him_on_policy_runner.py'):
            extra='from rsl_rl.algorithms.him_right_rear_split_ppo import HIMRightRearSplitPPO\n'
            assert current.read_text().count(extra)==1
            assert current.read_text().replace(extra,'')==archived.read_text()
        else:assert hashlib.sha256(current.read_bytes()).hexdigest()==digest,name
    spec=json.dumps(json.loads((ROOT/'.aris/compute/local-env-spec.json').read_text()),sort_keys=True,separators=(',',':'))
    assert hashlib.sha256(spec.encode()).hexdigest()[:8]=='fbc8f0c4'

def check_weighting():
    import torch
    from rsl_rl.algorithms.him_right_rear_split_ppo import HIMRightRearSplitPPO
    from rsl_rl.algorithms.him_rear_teacher_ppo import HIMRearTeacherPPO
    obj=object.__new__(HIMRightRearSplitPPO)
    obj.rear_teacher_coef=.1;obj.rear_teacher_forward_coef=.3;obj.other_legs_coef=0.
    obj.teacher_stand_threshold=.05;obj.teacher_forward_threshold=.1;obj.teacher_other_command_tolerance=.05
    obj.teacher_max_samples=4096;obj.command_scale=torch.tensor([2.,2.,.25]);obj.teacher=object()
    obj.rear_indices=torch.tensor([3,4,5]);obj.other_indices=torch.tensor([0,1,2,6,7,8,9,10,11])
    obj._split_totals=[0.,0.,0,0]
    obj.teacher_targets=lambda obs:(torch.zeros(len(obs),3),torch.zeros(len(obs),9))
    obs=torch.zeros(4,270);obs[1,0]=.7;obs[2,1]=.7;obs[3,2]=.1
    mu=torch.ones(4,12,requires_grad=True)
    loss,stats=obj.distillation_loss(obs,mu);loss.backward()
    assert abs(loss.item()-.2)<1e-6 and stats['samples']==2
    assert torch.allclose(mu.grad[1,3:6],mu.grad[0,3:6]*3)
    assert not mu.grad[:,obj.other_indices].any() and not mu.grad[2:].any()
    obj.rear_teacher_forward_coef=.1
    old,_=HIMRearTeacherPPO.distillation_loss(obj,obs,mu)
    new,_=obj.distillation_loss(obs,mu);assert torch.allclose(old,new)
    return dict(stand_coefficient=.1,forward_coefficient=.3,forward_gradient_multiplier=3,
                equal_weights_reproduce_previous_loss=True,other_nine_outputs_no_teacher_gradient=True,
                other_commands_not_supervised=True,training_updates=0)

def equal_tree(a,b):
    import torch
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def restore(runner):
    import torch
    saved=torch.load(RESUME,map_location=runner.device);assert saved['iter']==300
    runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
    runner.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
    runner.alg.actor_critic.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
    runner.alg.learning_rate=saved['optimizer_state_dict']['param_groups'][0]['lr']
    runner.current_learning_iteration=300;runner.alg.assert_frozen_policy()
    assert equal_tree(runner.alg.actor_critic.state_dict(),saved['model_state_dict'])
    assert equal_tree(runner.alg.optimizer.state_dict(),saved['optimizer_state_dict'])
    assert equal_tree(runner.alg.actor_critic.estimator.optimizer.state_dict(),saved['estimator_optimizer_state_dict'])
    base=torch.load(SOURCE,map_location=runner.device)['model_state_dict']
    frozen=[k for k in base if k=='std' or k.startswith(('actor.','estimator.'))]
    assert all(torch.equal(base[k],saved['model_state_dict'][k]) for k in frozen)
    return dict(model_exactly_restored=True,optimizer_exactly_restored=True,estimator_optimizer_exactly_restored=True,
                frozen_tensors_unchanged=len(frozen),source_iteration=300,learning_rate=runner.alg.learning_rate)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preflight',action='store_true');p.add_argument('--iterations',type=int);p.add_argument('--gpu');p.add_argument('--run-dir',type=Path);args=p.parse_args()
    preflight()
    if args.gpu:os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import isaacgym
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import HIMOnPolicyRunner
    checks=check_weighting()
    cfg,train=task_registry.get_cfgs(TASK)
    old_env=json.loads((RESUME.parent/'env_config.json').read_text());old_env['domain_rand'].pop('push_interval',None)
    assert class_to_dict(cfg)==old_env
    old_train=json.loads((RESUME.parent/'train_config.json').read_text())
    delta=differences(old_train,class_to_dict(train))
    allowed={'algorithm.rear_teacher_forward_coef','runner.algorithm_class_name','runner.experiment_name','runner.max_iterations'}
    assert set(delta)<=allowed,delta
    assert train.algorithm.rear_teacher_coef==.1 and train.algorithm.rear_teacher_forward_coef==.3
    assert cfg.env.num_envs==4096 and train.runner.num_steps_per_env==100 and train.runner.save_interval==10
    if args.preflight:
        print('SPLIT_PREFLIGHT_OK',json.dumps(dict(checks=checks,config_changes=delta)),flush=True);return
    if args.iterations!=300 or not args.gpu or not args.run_dir:p.error('This authorized run requires --iterations 300, --gpu and a fresh --run-dir')
    out=args.run_dir.resolve();logroot=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v3_right_rear_split'
    assert not out.exists() and logroot in out.parents
    assert torch.cuda.is_available()
    target=300+args.iterations;train.runner.max_iterations=target
    sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed',str(train.seed),'--sim_device','cuda:0','--rl_device','cuda:0']
    sim_args=get_args();out.mkdir(parents=True,exist_ok=False);env=runner=None
    try:
        write_json(out/'status.json',dict(state='initializing',completed_updates=300,target_updates=target,additional_completed=0))
        env,_=task_registry.make_env(TASK,args=sim_args,env_cfg=cfg)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),str(out),device='cuda:0')
        audit=restore(runner);attach_teacher(runner,env,train);env.take_diagnostics()
        write_json(out/'resume_audit.json',audit);write_json(out/'weighting_checks.json',checks)
        metadata=dict(task=TASK,version='new_v3_right_rear_split',source_checkpoint=str(RESUME),source_sha256=RESUME_SHA256,
            teacher_checkpoint=str(SOURCE),teacher_sha256=SOURCE_SHA256,teacher_frozen=True,baseline_actor_estimator_std_frozen=True,
            optimizer_initialization='Restored complete RH-only300 optimizer moments and saved LR',source_iteration=300,start_iteration=300,
            additional_iterations=args.iterations,target_iterations=target,initial_learning_rate=runner.alg.learning_rate,
            save_interval=10,auto_stop=False,gpu=args.gpu,pid=os.getpid(),seed=train.seed,distillation=class_to_dict(train.algorithm),
            continuation_scope='Weights and optimizer restored; simulator episodes and RNG restarted with original seed; not a bitwise uninterrupted run')
        sources=set(json.loads((RESUME.parent/'run_metadata.json').read_text())['code_sha256'])|set(NEW_SOURCES)
        metadata['code_sha256']={}
        for name in sorted(sources):
            source=ROOT/name;dest=out/'provenance'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            metadata['code_sha256'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
        for name,value in [('run_metadata',metadata),('env_config',class_to_dict(cfg)),('train_config',class_to_dict(train)),('config_diff_from_rh300',dict(env={},training=differences(old_train,class_to_dict(train))))]:write_json(out/(name+'.json'),value)
        def log_distillation(completed):
            metrics=runner.alg.distillation_metrics
            with (out/'distillation_metrics.jsonl').open('a') as stream:stream.write(json.dumps(dict(update=completed,**metrics),allow_nan=False)+'\n')
            for key,value in metrics.items():runner.writer.add_scalar('Distillation/'+key,value,completed)
            status=json.loads((out/'status.json').read_text());status.update(additional_completed=completed-300,additional_target=args.iterations);write_json(out/'status.json',status)
        install_logging(runner,out,metadata,target,after_update=log_distillation)
        print('SPLIT_RESUME_OK',json.dumps(audit),flush=True)
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==target
        final=torch.load(out/('model_%d.pt'%target),map_location='cpu');assert final['iter']==target and all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
        assert hashlib.sha256(RESUME.read_bytes()).hexdigest()==RESUME_SHA256
        write_json(out/'status.json',dict(state='completed',completed_updates=target,target_updates=target,additional_completed=args.iterations,additional_target=args.iterations))
    except BaseException as exc:
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        status.update(state='stopped_by_user' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));write_json(out/'status.json',status);raise
    finally:
        if runner is not None and runner.writer is not None:runner.writer.close()
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
