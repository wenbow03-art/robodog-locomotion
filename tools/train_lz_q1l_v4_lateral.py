"""Review-ready V4 launcher. Requires an explicit later-approved training budget."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
TASK='lz_q1l_flat_v4_lateral'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,required=True)
    parser.add_argument('--gpu',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    opts=parser.parse_args()
    if opts.iterations<=0:parser.error('Explicit positive new update count required')
    out=opts.run_dir.resolve();parent=ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_v4_lateral'
    if parent not in out.parents or out.exists():parser.error('Use a NEW run directory under logs2/flat_lz_q1l_v4_lateral')
    os.environ['CUDA_VISIBLE_DEVICES']=opts.gpu
    import isaacgym
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils import get_args
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.runners.him_lateral_runner import HIMLateralRunner
    from train_lz_q1l_v2_unified import install_logging
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; training not started')
    sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
    args=get_args();cfg,train=task_registry.get_cfgs(TASK);train.runner.max_iterations=opts.iterations
    out.mkdir(parents=True);env=runner=None
    def write(name,data):(out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    try:
        write('status',dict(state='initializing',target_updates=opts.iterations))
        env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMLateralRunner(env,class_to_dict(train),str(out),device=args.rl_device)
        model=runner.alg.actor_critic
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith(('critic.','lateral_delta.')) and k!='lateral_log_std'}
        sources=[Path(__file__).resolve()]+list((ROOT/'HIMLoco/legged_gym/legged_gym/envs/lz_q1l').glob('*.py'))
        sources += [ROOT/p for p in ['HIMLoco/legged_gym/legged_gym/utils/lz_q1l_stand_walk.py','HIMLoco/legged_gym/legged_gym/utils/lz_q1l_lateral.py','HIMLoco/legged_gym/legged_gym/utils/lz_q1l_mirror.py','HIMLoco/rsl_rl/rsl_rl/modules/him_lateral.py','HIMLoco/rsl_rl/rsl_rl/modules/him_right_rear_only.py','HIMLoco/rsl_rl/rsl_rl/algorithms/him_lateral_ppo.py','HIMLoco/rsl_rl/rsl_rl/runners/him_lateral_runner.py','HIMLoco/rsl_rl/rsl_rl/runners/him_on_policy_runner.py','HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py','HIMLoco/legged_gym/legged_gym/envs/__init__.py','HIMLoco/legged_gym/legged_gym/utils/task_registry.py','tools/train_lz_q1l_v2_unified.py']]
        hashes={}
        for source in sources:
            rel=source.relative_to(ROOT);dest=out/'provenance'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);hashes[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
        metadata=dict(task=TASK,baseline=train.runner.baseline_directory,baseline_sha256='23d99708723bacb0a5cc5c473a66e76f076a3444a738a1a27c1a9f9d6b937086',
            additional_iterations=opts.iterations,save_interval=50,auto_stop=False,gpu=opts.gpu,pid=os.getpid(),code_sha256=hashes,
            initialization='Frozen new baseline; zero lateral residual, fresh lateral PPO/critic optimizer. No old V7/V8 weights or LH-to-RH teacher.',
            metrics='Direction-separated direct controller rollout, no mirroring; standing/forward deterministic, lateral exploratory.',
            lateral_direction_sampling='Balanced 50/50 command counts; odd batches alternate the extra sign. Exposure time can differ after resets.')
        for name,value in [('run_metadata',metadata),('env_config',class_to_dict(cfg)),('train_config',class_to_dict(train))]:write(name,value)
        def audit(step):
            state=model.state_dict()
            assert all(torch.equal(state[k].cpu(),v) for k,v in frozen.items()),'Frozen baseline changed'
            write('frozen_audit',dict(completed_updates=step,unchanged=True,tensors=len(frozen)))
        install_logging(runner,out,metadata,opts.iterations,after_update=audit)
        env.take_diagnostics();runner.learn(opts.iterations,init_at_random_ep_len=True)
        audit(opts.iterations);write('status',dict(state='completed',completed_updates=opts.iterations,target_updates=opts.iterations))
    except BaseException as exc:
        write('status',dict(state='stopped_by_user' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc)));raise
    finally:
        if runner is not None and runner.writer is not None:runner.writer.close()
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
