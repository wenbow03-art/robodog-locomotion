"""New v3 training entrypoint. Review first; no default training launch/count."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from train_lz_q1l_v1_support_500 import differences, write_json
from train_lz_q1l_v2_unified import ROOT, install_logging

TASK = 'lz_q1l_flat_v3_rear_teacher'
SOURCE = ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v2_gait/gait_from_posture500_200_20261025_113614_3235946/model_150.pt'
SOURCE_SHA256 = '655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747'
REVIEW = ROOT/'media/v3_rear_teacher_sweep_review_20261026'
NEW_SOURCES = ['tools/train_lz_q1l_v3_rear_teacher.py', 'tools/check_lz_q1l_v3_rear_teacher.py',
    'tools/check_lz_q1l_v3_rear_teacher.sh', 'HIMLoco/rsl_rl/rsl_rl/algorithms/him_rear_teacher_ppo.py',
    'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py',
    'tools/queue_lz_q1l_v3_sweep.py', 'tools/start_lz_q1l_v3_sweep.sh',
    'tools/eval_lz_q1l_v3_sweep.py', 'tools/audit_lz_q1l_new_lineage.py',
    'tools/summarize_lz_q1l_lineage_audit.py']


def verify_baseline():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    parent=json.loads((SOURCE.parent/'run_metadata.json').read_text())
    for name,digest in parent['code_sha256'].items():
        if name.endswith('/envs/__init__.py'):
            continue
        archived=SOURCE.parent/'provenance'/name
        current=ROOT/name
        if name.endswith(('/lz_q1l_2.py','/lz_q1l_2_config.py')):
            assert hashlib.sha256(archived.read_bytes()).hexdigest()==digest
            assert current.read_bytes().startswith(archived.read_bytes()),name
        elif name.endswith('/him_on_policy_runner.py'):
            assert hashlib.sha256(archived.read_bytes()).hexdigest()==digest
            added='from rsl_rl.algorithms.him_rear_teacher_ppo import HIMRearTeacherPPO\n'
            assert current.read_text().count(added)==1
            assert current.read_text().replace(added,'')==archived.read_text(),name
        else:
            assert hashlib.sha256(current.read_bytes()).hexdigest()==digest,name


def attach_teacher(runner, env, train_cfg):
    """Teacher always comes from pinned baseline, including on future resumes."""
    import torch
    from rsl_rl.modules import HIMActorCritic
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
    teacher=HIMActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,
                          env.num_actions,**class_to_dict(train_cfg.policy)).to(runner.device)
    saved=torch.load(SOURCE,map_location=runner.device)
    assert saved['iter']==150
    teacher.load_state_dict(saved['model_state_dict'],strict=True)
    runner.alg.set_teacher(teacher,LZQ1LMirror.from_env(env,runner.device),env.dof_names,env.commands_scale)


def preflight():
    verify_baseline()
    report=json.loads((REVIEW/'report.json').read_text())
    assert report['training_updates']==0 and report['parameters_unchanged']
    assert report['checkpoint_sha256']==SOURCE_SHA256
    assert all(report['checks'].values())
    for source,digest in report['source_sha256'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest,source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--iterations',type=int)
    parser.add_argument('--gpu')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--other-legs-coef',type=float,choices=[0.,.01,.02,.03,.04],default=.02)
    args=parser.parse_args()
    preflight()
    if args.preflight:
        print('V3_PREFLIGHT_OK no training started',flush=True)
        return
    if not args.iterations or args.iterations<1 or not args.gpu or not args.run_dir:
        parser.error('Explicit positive --iterations, --gpu and fresh --run-dir required')
    out=args.run_dir.resolve()
    logroot=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v3_rear_teacher'
    if out.exists() or logroot.resolve() not in out.parents:
        parser.error('Use a new run directory under logs2/flat_lz_q1l_v3_rear_teacher')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    import isaacgym
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners import HIMOnPolicyRunner
    assert torch.cuda.is_available()
    cfg,train=task_registry.get_cfgs(TASK)
    train.algorithm.other_legs_coef=args.other_legs_coef
    archived=json.loads((SOURCE.parent/'env_config.json').read_text())
    archived['domain_rand'].pop('push_interval',None)
    assert class_to_dict(cfg)==archived
    assert cfg.env.num_envs==4096 and train.runner.num_steps_per_env==100
    train.runner.max_iterations=args.iterations
    sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed',str(train.seed),
              '--sim_device','cuda:0','--rl_device','cuda:0']
    sim_args=get_args();out.mkdir(parents=True,exist_ok=False)
    env=runner=None
    try:
        write_json(out/'status.json',dict(state='initializing',completed_updates=0,target_updates=args.iterations))
        env,_=task_registry.make_env(TASK,args=sim_args,env_cfg=cfg)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),str(out),device='cuda:0')
        runner.load(str(SOURCE),load_optimizer=True)
        assert runner.current_learning_iteration==150
        assert runner.alg.optimizer.state and runner.alg.actor_critic.estimator.optimizer.state
        runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]['lr']
        attach_teacher(runner,env,train)
        runner.current_learning_iteration=0;env.take_diagnostics()
        metadata=dict(task=TASK,version='new_v3',source_checkpoint=str(SOURCE),source_sha256=SOURCE_SHA256,
            teacher_checkpoint=str(SOURCE),teacher_sha256=SOURCE_SHA256,teacher_frozen=True,
            source_iteration=150,start_iteration=0,target_iterations=args.iterations,
            initial_learning_rate=runner.alg.learning_rate,save_interval=10,auto_stop=False,
            gpu=args.gpu,pid=os.getpid(),seed=train.seed,distillation=class_to_dict(train.algorithm))
        sources=set(json.loads((SOURCE.parent/'run_metadata.json').read_text())['code_sha256'])|set(NEW_SOURCES)
        metadata['code_sha256']={}
        for name in sorted(sources):
            source=ROOT/name;target=out/'provenance'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target);metadata['code_sha256'][name]=hashlib.sha256(source.read_bytes()).hexdigest()
        for name,value in [('run_metadata',metadata),('env_config',class_to_dict(cfg)),('train_config',class_to_dict(train)),
                           ('config_diff_from_gait150',dict(env={},training=differences(json.loads((SOURCE.parent/'train_config.json').read_text()),class_to_dict(train))))]:
            write_json(out/(name+'.json'),value)
        def log_distillation(completed):
            metrics=runner.alg.distillation_metrics
            with (out/'distillation_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=completed,**metrics),allow_nan=False)+'\n')
            for key,value in metrics.items():runner.writer.add_scalar('Distillation/'+key,value,completed)
        install_logging(runner,out,metadata,args.iterations,after_update=log_distillation)
        print('V3_INITIALIZATION_OK',json.dumps(metadata),flush=True)
        runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==args.iterations
        final=torch.load(out/('model_%d.pt'%args.iterations),map_location='cpu')
        assert final['iter']==args.iterations and all(torch.isfinite(v).all() for v in final['model_state_dict'].values())
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
        write_json(out/'status.json',dict(state='completed',completed_updates=args.iterations,target_updates=args.iterations))
    except BaseException as exc:
        status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        status.update(state='stopped_by_user' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        write_json(out/'status.json',status);raise
    finally:
        if runner is not None and runner.writer is not None:runner.writer.close()
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
