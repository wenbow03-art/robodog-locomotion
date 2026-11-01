"""Standing/forward repair with checkpoint continuation and log-only posture checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from train_lz_q1l_v1_support_500 import differences, write_json
from train_lz_q1l_v2_unified import ROOT, SOURCE, SOURCE_SHA256, load_new_v1_weights, install_logging

TASK='lz_q1l_flat_v2_posture'


def regression_reasons(baseline, report):
    """Monitoring thresholds, not additional learning rewards."""
    reasons=[]
    for name,before in baseline['cases'].items():
        row=report['cases'][name]
        if 'height_mean_m' not in row:
            reasons.append(name+': no valid post-settle samples')
            continue
        if row['height_mean_m'] < max(.18, before['height_mean_m']-.03):
            reasons.append('%s: height %.3f m below baseline %.3f m / 0.18 m floor'%(name,row['height_mean_m'],before['height_mean_m']))
        if row['failures'] > before['failures']+1:
            reasons.append(name+': failures increased by more than one of three trials')
        for i,(old,new) in enumerate(zip(before['hard_limit_rate_per_joint'],row['hard_limit_rate_per_joint'])):
            if new>.5 and new>old+.3:
                reasons.append('%s: joint %d at hard limit %.1f%% (baseline %.1f%%)'%(name,i,100*new,100*old))
    return reasons


def evaluate(checkpoint,out):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    with out.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'tools/eval_lz_q1l_posture.py'),
            '--checkpoint',str(checkpoint),'--out',str(out)],cwd=str(ROOT),env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    return json.loads((out/'report.json').read_text())


def record_posture_check(runner,out,baseline,completed):
    """Quality findings and evaluation failures never request a training stop."""
    if completed%20:return
    status=json.loads((out/'status.json').read_text())
    status['state']='evaluating_posture';write_json(out/'status.json',status)
    result=dict(update=completed,auto_stop=False,stop=False)
    try:
        report=evaluate(out/('model_%d.pt'%completed),out/('eval_%03d'%completed))
        reasons=regression_reasons(baseline,report)
        result.update(evaluation_ok=True,regression_detected=bool(reasons),reasons=reasons,cases=report['cases'])
        for name,row in report['cases'].items():
            if 'height_mean_m' in row:
                runner.writer.add_scalar('PostureEval/'+name+'/height_m',row['height_mean_m'],completed)
                runner.writer.add_scalar('PostureEval/'+name+'/vx',row['vx'],completed)
        runner.writer.flush()
    except Exception as exc:
        result.update(evaluation_ok=False,error=repr(exc))
    write_json(out/('posture_check_%03d.json'%completed),result)
    print('POSTURE_CHECK_LOG_ONLY',completed,json.dumps(result),flush=True)
    status.update(state='running',last_posture_check=completed,last_posture_check_ok=result['evaluation_ok'])
    write_json(out/'status.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True,help='Verified CPU source-policy preflight report')
    parser.add_argument('--iterations',type=int,default=500,help='Total completed updates, including the resumed checkpoint')
    parser.add_argument('--resume',type=Path,help='Posture checkpoint with both optimizer states')
    opts=parser.parse_args()
    baseline=json.loads(opts.baseline.read_text())
    assert baseline['checkpoint_sha256']==SOURCE_SHA256 and baseline['iteration']==500
    assert baseline['rule_checks']['command_draws']==9000
    for path,digest in baseline['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    assert all('height_mean_m' in c for c in baseline['cases'].values())
    out=opts.run_dir.resolve()
    logroot=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_posture'
    if logroot.resolve() not in out.parents or out.exists():parser.error('Run must be a new directory under logs2/flat_lz_q1l_v2_posture')
    os.environ['CUDA_VISIBLE_DEVICES']=opts.gpu
    import isaacgym
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import HIMOnPolicyRunner
    assert torch.cuda.is_available()
    cfg,train=task_registry.get_cfgs(TASK)
    old,old_train=task_registry.get_cfgs('lz_q1l_flat_v2_unified')
    assert opts.iterations>0 and train.runner.save_interval==10
    train.runner.max_iterations=opts.iterations
    assert cfg.env.num_envs==4096 and train.runner.num_steps_per_env==100
    source=opts.resume.resolve() if opts.resume else SOURCE
    source_digest=hashlib.sha256(source.read_bytes()).hexdigest()
    start=0
    if opts.resume:
        parent=json.loads((source.parent/'run_metadata.json').read_text())
        assert parent['task']==TASK
        parent_env=json.loads((source.parent/'env_config.json').read_text())
        env_diff=differences(parent_env,class_to_dict(cfg))
        # The simulator derives push_interval (in policy steps) during construction.
        assert set(env_diff)<= {'domain_rand.push_interval'},env_diff
        train_diff=differences(json.loads((source.parent/'train_config.json').read_text()),class_to_dict(train))
        assert set(train_diff)<= {'runner.max_iterations'},train_diff
        state=torch.load(source,map_location='cpu')
        start=int(state['iter'])
        assert 0<start<opts.iterations
        assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        assert state['optimizer_state_dict']['state'] and state['estimator_optimizer_state_dict']['state']
        del state
    sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed',str(train.seed),'--sim_device','cuda:0','--rl_device','cuda:0']
    args=get_args();out.mkdir(parents=True,exist_ok=False)
    env=runner=None
    try:
        write_json(out/'status.json',dict(state='initializing',completed_updates=start,target_updates=opts.iterations))
        write_json(out/'config_diff_from_six_command_v2.json',dict(env=differences(class_to_dict(old),class_to_dict(cfg)),
            training=differences(class_to_dict(old_train),class_to_dict(train))))
        shutil.copytree(opts.baseline.parent,out/'eval_000')
        env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        assert env.device=='cuda:0' and env.num_envs==4096
        if opts.resume:
            assert class_to_dict(cfg)==parent_env
        runner=HIMOnPolicyRunner(env,class_to_dict(train),str(out),device=args.rl_device)
        if opts.resume:
            runner.load(str(source),load_optimizer=True)
            assert runner.current_learning_iteration==start
            rates={group['lr'] for group in runner.alg.optimizer.param_groups}
            assert len(rates)==1
            runner.alg.learning_rate=rates.pop()
            assert runner.alg.optimizer.state and runner.alg.actor_critic.estimator.optimizer.state
        else:
            load_new_v1_weights(runner)
        env.take_diagnostics()
        print('INITIALIZATION_OK',dict(source=str(source),completed_updates=start,target_updates=opts.iterations,
            optimizer_restored=bool(opts.resume),learning_rate=runner.alg.learning_rate),flush=True)
        metadata=dict(task=TASK,source_checkpoint=str(source),source_sha256=source_digest,
            initialization=('Resume full weights, PPO and estimator optimizers and PPO learning rate; simulation and RNG restarted'
                if opts.resume else 'Full new V1 weights; fresh PPO and estimator optimizers'),
            start_iteration=start,target_iterations=opts.iterations,additional_iterations=opts.iterations-start,
            save_interval=10,checkpoint_iterations=sorted(set(list(range((start//10+1)*10,opts.iterations+1,10))+[opts.iterations])),
            evaluation_interval=20,evaluation_backend='CPU PhysX with paired starts, baseline from same protocol',
            posture_monitor=dict(auto_stop=False,height_drop_from_baseline_m=.03,minimum_mean_height_m=.18,
                hard_limit_min_fraction=.5,hard_limit_increase_fraction=.3,allowed_added_failures=1),
            gpu=opts.gpu,pid=os.getpid(),seed=train.seed,
            feet=[env.gym.get_actor_rigid_body_names(env.envs[0],env.actor_handles[0])[i] for i in env.feet_indices.tolist()])
        sources=[Path(__file__).resolve(),ROOT/'tools/eval_lz_q1l_posture.py',ROOT/'tools/start_lz_q1l_v2_posture_500.sh',
            ROOT/'tools/train_lz_q1l_v2_unified.py',ROOT/'tools/train_lz_q1l_v1_support_500.py']
        sources += [ROOT/p for p in (
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py','HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
            'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py','HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
            'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py','HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
            'HIMLoco/legged_gym/legged_gym/envs/__init__.py','HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
            'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py','HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py',
            'HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py','HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf')]
        metadata['code_sha256']={}
        for source in sources:
            relative=source.relative_to(ROOT);target=out/'provenance'/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            metadata['code_sha256'][str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
        for name,value in [('run_metadata',metadata),('env_config',class_to_dict(cfg)),('train_config',class_to_dict(train))]:write_json(out/(name+'.json'),value)

        def after_update(completed):
            record_posture_check(runner,out,baseline,completed)

        install_logging(runner,out,metadata,opts.iterations,after_update=after_update)
        runner.learn(num_learning_iterations=opts.iterations-start,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==opts.iterations
        for iteration in metadata['checkpoint_iterations']:
            state=torch.load(out/('model_%d.pt'%iteration),map_location='cpu')
            assert state['iter']==iteration and all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
        write_json(out/'status.json',dict(state='completed',completed_updates=opts.iterations,target_updates=opts.iterations,
            last_posture_check=(opts.iterations//20)*20))
    except BaseException as exc:
        status=json.loads((out/'status.json').read_text());status.update(state='stopped_by_user' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        write_json(out/'status.json',status);raise
    finally:
        if runner is not None and runner.writer is not None:runner.writer.close()
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
