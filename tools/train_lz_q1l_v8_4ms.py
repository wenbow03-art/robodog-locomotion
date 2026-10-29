"""Adapt original V8 to4ms physics for200 updates, without cadence reward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'HIMLoco/legged_gym/logs1/flat_lz_q1l_v8/v8_ft_v7_200_200iters_20261018_130426_2712609/model_200.pt'
SHA = '3480ca2abddfcd62df2be30d89e53a992e3eadc100add59df131213feef48a5a'
TASK = 'lz_q1l_flat_v8_4ms'
FILES = [
    'tools/train_lz_q1l_v8_4ms.py',
    'HIMLoco/legged_gym/legged_gym/envs/__init__.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_2_config.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l.py',
    'HIMLoco/legged_gym/legged_gym/envs/lz_q1l/lz_q1l_config.py',
    'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py',
    'HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py',
    'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_cadence.py',
    'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py',
    'HIMLoco/legged_gym/resources/robots/lz_q1l/urdf/lz_q1l.urdf',
    'HIMLoco/rsl_rl/rsl_rl/modules/him_actor_critic.py',
    'HIMLoco/rsl_rl/rsl_rl/modules/him_estimator.py',
    'HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
    'HIMLoco/rsl_rl/rsl_rl/algorithms/him_ppo.py',
]


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a)|set(b)) for p in differences(a.get(k), b.get(k), prefix+'.'+k)]
    return [] if a == b else [prefix]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    opts=parser.parse_args();out=opts.run_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(state='initializing',completed_updates=0,target_updates=200))
    env=runner=None
    try:
        import isaacgym
        import torch
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        from legged_gym.utils.helpers import class_to_dict
        from rsl_rl.runners import HIMOnPolicyRunner
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
        baseline=ROOT/'HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027/model_800.pt'
        baseline_sha=hashlib.sha256(baseline.read_bytes()).hexdigest()
        assert baseline_sha=='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086'
        cfg,train=task_registry.get_cfgs(TASK)
        old=json.loads(SOURCE.with_name('env_config.json').read_text())
        old['domain_rand'].pop('push_interval',None)
        changed=differences(old,class_to_dict(cfg))
        assert set(changed)=={'.sim.dt','.control.decimation','.domain_rand.action_delay_steps'},changed
        old_train=json.loads(SOURCE.with_name('train_config.json').read_text())
        assert old_train['policy']==class_to_dict(train)['policy']
        assert set(differences(old_train['algorithm'],class_to_dict(train)['algorithm']))=={'.learning_rate','.schedule'}
        assert train.runner.num_steps_per_env==old_train['runner']['num_steps_per_env']==100
        assert cfg.sim.dt==.004 and cfg.control.decimation==5
        assert class_to_dict(cfg)['rewards']==old['rewards']
        metadata=dict(task=TASK,source_checkpoint=str(SOURCE),source_sha256=SHA,
            physics_dt=.004,decimation=5,additional_iterations=200,save_interval=50,
            iteration_numbering='New4ms adaptation updates0..200; source is oldV8 model200',
            baseline_checkpoint=str(baseline),baseline_sha256=baseline_sha,
            load_optimizer=True,seed=1,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
            configuration_changes=changed,code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},
            mirroring='Inherited oldV8 native left/right training; playback left mirrors right. Baseline is separate and never loaded into this optimizer.')
        for name in FILES:
            dest=out/'provenance'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        write(out/'run_metadata.json',metadata)
        write(out/'env_config.json',class_to_dict(cfg));write(out/'train_config.json',class_to_dict(train))
        sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
        args=get_args()
        if opts.preflight:cfg.env.num_envs=args.num_envs=32
        env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),None if opts.preflight else str(out),device=env.device)
        saved=torch.load(SOURCE,map_location=env.device);assert saved['iter']==200
        runner.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        runner.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
        estimator=runner.alg.actor_critic.estimator
        estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
        runner.alg.learning_rate=1e-5
        for group in runner.alg.optimizer.param_groups:group['lr']=1e-5
        estimator.learning_rate=estimator.optimizer.param_groups[0]['lr']
        assert all(torch.equal(v,runner.alg.actor_critic.state_dict()[k]) for k,v in saved['model_state_dict'].items())
        write(out/'resume_audit.json',dict(exact_model=True,optimizer_restored=True,
            optimizer_learning_rate=runner.alg.learning_rate,estimator_learning_rate=estimator.learning_rate))
        del saved
        if opts.preflight:
            obs,_=env.reset();model=runner.alg.actor_critic
            for step in range(160):
                env.commands.zero_();env.commands[:16,1]=.3;env.commands[16:24,1]=-.3;env.commands[24:,0]=.35
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                with torch.no_grad():actions=model.act_inference(obs)
                obs,_,reward,done,_,_,_=env.step(actions)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                assert not hasattr(env,'cycle_penalty')
            write(out/'checks.json',dict(passed=True,original_rewards=True,finite=True,
                envs=32,steps=160,physics_dt=env.cfg.sim.dt,decimation=env.cfg.control.decimation,
                code_sha256=metadata['code_sha256']))
            write(out/'status.json',dict(state='preflight_passed',completed_updates=0,target_updates=200))
            print('GPU_PREFLIGHT_PASS',flush=True);return
        runner.current_learning_iteration=0
        original_log,original_save=runner.log,runner.save
        original_save(str(out/'model_0.pt'),infos=metadata)
        def log(locs):
            original_log(locs);completed=locs['it']+1
            if completed%50==0:
                old_iteration=runner.current_learning_iteration;runner.current_learning_iteration=completed
                try:
                    path=out/('model_%d.pt'%completed);assert not path.exists()
                    original_save(str(path)+'.tmp',infos=metadata);os.replace(str(path)+'.tmp',path)
                finally:runner.current_learning_iteration=old_iteration
                assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_sha
            write(out/'status.json',dict(state='running',completed_updates=completed,target_updates=200,
                iteration_seconds=locs['collection_time']+locs['learn_time'],pid=os.getpid()))
        runner.log=log;runner.save=lambda *a,**k:None
        runner.learn(num_learning_iterations=200,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==200
        for step in (50,100,150,200):assert (out/('model_%d.pt'%step)).is_file()
        runner.writer.flush();runner.writer.close()
        assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_sha
        write(out/'status.json',dict(state='completed',completed_updates=200,target_updates=200))
        print('TRAINING_COMPLETE',flush=True)
    except BaseException as exc:
        state=json.loads((out/'status.json').read_text());state.update(state='failed',error=repr(exc))
        write(out/'status.json',state);raise
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
