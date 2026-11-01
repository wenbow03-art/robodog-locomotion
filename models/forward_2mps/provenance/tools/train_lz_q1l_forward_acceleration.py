"""Explicit-budget acceleration continuation. --preflight updates are discarded."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from lz_q1l_acceleration_common import ROOT,TASK,SOURCE,SOURCE_SHA,STANDING,STANDING_SHA,attach_standing,digest,write
from train_lz_q1l_stance_width import same

EXTRA_FILES = [
    'tools/train_lz_q1l_forward_acceleration.py', 'tools/lz_q1l_acceleration_common.py',
    'tools/evaluate_lz_q1l_forward_acceleration.py', 'tools/check_lz_q1l_forward_acceleration.py',
    'tools/run_lz_q1l_forward_acceleration.sh', 'tools/launch_lz_q1l_forward_acceleration.py',
    'HIMLoco/legged_gym/legged_gym/utils/lz_q1l_acceleration.py',
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--iterations',type=int)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--resume',type=Path)
    opt=p.parse_args()
    if not opt.preflight and (opt.iterations is None or opt.iterations<=0):p.error('Explicit positive --iterations required')
    if opt.preflight and (opt.iterations is not None or opt.resume):p.error('Preflight has exactly2 discarded updates, no resume')
    updates=2 if opt.preflight else opt.iterations
    out=opt.run_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(state='initializing',preflight=opt.preflight,requested_updates=updates))
    env=None
    try:
        assert digest(SOURCE)==SOURCE_SHA and digest(STANDING)==STANDING_SHA
        import isaacgym
        import torch
        from legged_gym.envs import task_registry
        from legged_gym.utils import get_args
        from legged_gym.utils.helpers import class_to_dict
        from legged_gym.utils.lz_q1l_acceleration import AccelerationCurriculum
        from rsl_rl.runners.him_ema_rear_teacher_runner import HIMEmaRearTeacherRunner
        from train_lz_q1l_v1_support_500 import differences
        cfg,train=task_registry.get_cfgs(TASK)
        old=json.loads((SOURCE.parent/'env_config.json').read_text());old['domain_rand'].pop('push_interval',None)
        changes=differences(old,class_to_dict(cfg))
        assert set(changes)=={'commands.resampling_time','acceleration','rewards.forward_yaw_min_speed','rewards.forward_yaw_command_tolerance','rewards.scales.forward_yaw_rate'},changes
        old_train=json.loads((SOURCE.parent/'train_config.json').read_text())
        tc=differences(old_train,class_to_dict(train))
        assert set(tc)=={'runner.experiment_name','runner.max_iterations'},tc
        assert cfg.sim.dt==.001 and cfg.control.decimation==20 and cfg.env.num_envs==4096
        assert cfg.commands.zero_command_prob==cfg.commands.pure_forward_prob==.5
        if opt.preflight:cfg.env.num_envs=128
        source=opt.resume.resolve() if opt.resume else SOURCE
        if opt.resume:
            previous=json.loads((source.parent/'run_metadata.json').read_text())
            assert previous['task']==TASK and not previous['preflight'],'Cannot train from a discarded preflight'
            assert previous['standing_sha256']==STANDING_SHA
        else:previous=json.loads((SOURCE.parent/'run_metadata.json').read_text())
        files=set(previous['code_sha256'])|set(EXTRA_FILES)
        metadata=dict(task=TASK,source_checkpoint=str(source),source_sha256=digest(source),
                      standing_checkpoint=str(STANDING),standing_sha256=STANDING_SHA,preflight=opt.preflight,
                      configuration_changes=changes,train_configuration_changes=tc,
                      requested_updates=updates,save_interval=50,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      code_sha256={f:digest(ROOT/f) for f in sorted(files)},
                      note='No automatic quality early stop. Gate failure retains stage until explicit update budget ends. Simulator/RNG restart on resume.')
        for f in files:
            dest=out/'provenance'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest)
        write(out/'run_metadata.json',metadata);write(out/'env_config.json',class_to_dict(cfg));write(out/'train_config.json',class_to_dict(train))
        sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cuda:0','--rl_device','cuda:0']
        args=get_args();env,_=task_registry.make_env(TASK,args=args,env_cfg=cfg)
        runner=HIMEmaRearTeacherRunner(env,class_to_dict(train),str(out),device=env.device)
        runner.load(str(source));saved=torch.load(source,map_location=env.device)
        model=runner.alg.actor_critic
        assert same(saved['model_state_dict'],model.state_dict())
        assert same(saved['teacher_state_dict'],runner.alg.teacher.state_dict())
        assert same(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict())
        assert same(saved['estimator_optimizer_state_dict'],model.estimator.optimizer.state_dict())
        teacher_start=runner.alg.teacher_updates
        gate=AccelerationCurriculum(cfg.acceleration)
        if opt.resume:
            gate.load_state_dict(saved['acceleration_curriculum'])
            start=saved['iter']
        else:
            assert saved['iter']==300 and teacher_start==600
            start=0
            runner.alg.learning_rate=1e-5
            for group in runner.alg.optimizer.param_groups:group['lr']=1e-5
        runner.current_learning_iteration=start
        env.set_acceleration_stage(gate.stage)
        standing=attach_standing(env)
        frozen={k:v.clone() for k,v in model.estimator.state_dict().items()}
        fixed_stand={k:v.clone() for k,v in standing.state_dict().items()}
        est_opt=copy.deepcopy(model.estimator.optimizer.state_dict())
        initial_actor={k:v.clone() for k,v in model.actor.state_dict().items()}
        initial_std=model.std.clone()
        write(out/'resume_audit.json',dict(exact_student=True,exact_teacher=True,teacher_start_updates=teacher_start,
              exact_both_optimizers=True,learning_rate=runner.alg.learning_rate,initial_stage=gate.stage))
        del saved
        original_save,original_log=runner.save,runner.log
        def checkpoint(path, completed):
            prior=runner.current_learning_iteration;runner.current_learning_iteration=completed
            temporary=Path(str(path)+'.tmp')
            try:original_save(str(temporary),infos=metadata)
            finally:runner.current_learning_iteration=prior
            state=torch.load(temporary,map_location=env.device)
            state['acceleration_curriculum']=gate.state_dict()
            torch.save(state,temporary);os.replace(str(temporary),str(path))
        checkpoint(out/('model_%d.pt'%start),start)
        checks={}
        if opt.preflight:
            from check_lz_q1l_forward_acceleration import check
            checks=check(env,runner,cfg)
        # No random episode-age initialization: it would skip the standing/ramp.
        with torch.inference_mode():env.reset()
        metrics=torch.zeros(7,device=env.device);metric_count=torch.zeros((),device=env.device)
        old_step=env.step
        def step(actions):
            result=old_step(actions)
            packet=env.ramp_step_metrics
            active=(packet[:,0]>.1)&~result[3].bool()
            values=torch.stack((packet[:,0],packet[:,1],(packet[:,1]-packet[:,0]).abs(),packet[:,3].abs(),
                                packet[:,5].abs()*180/torch.pi,packet[:,6].abs(),packet[:,3].square()*3),1)
            metrics.add_((values*active[:,None]).sum(0));metric_count.add_(active.sum())
            return result
        env.step=step
        def log(locs):
            original_log(locs);completed=locs['it']+1
            assert runner.alg.teacher_updates==teacher_start+completed-start
            assert same(frozen,model.estimator.state_dict()) and same(est_opt,model.estimator.optimizer.state_dict())
            assert same(fixed_stand,standing.state_dict()) and torch.equal(initial_std,model.std)
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            names=['command_vx','actual_vx','speed_mae','abs_yaw_rate','abs_heading_deg','abs_lateral_m','yaw_cost_before_dt']
            values={k:float(v) for k,v in zip(names,metrics/metric_count.clamp_min(1))}
            for k,v in values.items():runner.writer.add_scalar('Acceleration/'+k,v,completed)
            for k,v in env.pop_stance_width_metrics().items():runner.writer.add_scalar('StanceWidth/'+k,v,completed)
            for k,v in runner.alg.distillation_metrics.items():runner.writer.add_scalar('Teacher/'+k,v,completed)
            metrics.zero_();metric_count.zero_()
            decision=None
            if not opt.preflight and completed-gate.last_evaluation>=cfg.acceleration.evaluation_interval:
                eval_ck=out/('evaluation_model_%d.pt'%completed);checkpoint(eval_ck,completed)
                destination=out/('evaluation_%d'%completed)
                with (out/('evaluation_%d.log'%completed)).open('w') as stream:
                    try:
                        subprocess.run([sys.executable,str(ROOT/'tools/evaluate_lz_q1l_forward_acceleration.py'),
                            '--checkpoint',str(eval_ck),'--stage',str(gate.stage),'--out',str(destination)],
                            cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=600,check=True)
                        report=json.loads((destination/'report.json').read_text())
                        assert report['sha256']==digest(eval_ck)
                    except Exception as exc:
                        report=dict(error=repr(exc),groups=[])
                decision=gate.consider(completed,report)
                env.set_acceleration_stage(gate.stage)
                write(out/('gate_%d.json'%completed),dict(decision=decision,state=gate.state_dict(),report=report))
                print('ACCELERATION_GATE',completed,decision,flush=True)
            runner.writer.add_scalar('Acceleration/stage',gate.stage,completed)
            runner.writer.add_scalar('Acceleration/maximum_command_acceleration',gate.levels[gate.stage],completed)
            if completed%50==0 or completed==start+updates:
                checkpoint(out/('model_%d.pt'%completed),completed)
            write(out/'status.json',dict(state='running',preflight=opt.preflight,pid=os.getpid(),
                completed_updates=completed,target_updates=start+updates,additional_updates=completed-start,
                stage=gate.stage,maximum_acceleration=gate.levels[gate.stage],gate=decision,
                curriculum=gate.state_dict(),metrics=values,learning_rate=runner.alg.learning_rate))
        runner.log=log;runner.save=lambda *args,**kwargs:None
        runner.learn(updates,init_at_random_ep_len=False)
        assert any(not torch.equal(v,model.actor.state_dict()[k]) for k,v in initial_actor.items())
        assert digest(SOURCE)==SOURCE_SHA and digest(STANDING)==STANDING_SHA
        final=out/('model_%d.pt'%(start+updates));assert final.exists()
        state=torch.load(final,map_location=env.device)
        assert same(state['teacher_state_dict'],runner.alg.teacher.state_dict())
        assert state['acceleration_curriculum']==gate.state_dict()
        runner.load(str(final));gate.load_state_dict(state['acceleration_curriculum'])
        assert same(state['optimizer_state_dict'],runner.alg.optimizer.state_dict())
        runner.writer.flush();runner.writer.close()
        if opt.preflight:
            destination=out/'evaluation_smoke'
            with (out/'evaluation_smoke.log').open('w') as stream:
                subprocess.run([sys.executable,str(ROOT/'tools/evaluate_lz_q1l_forward_acceleration.py'),
                    '--checkpoint',str(final),'--stage','2','--out',str(destination)],
                    cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,timeout=600,check=True)
            report=json.loads((destination/'report.json').read_text())
            assert [g['acceleration'] for g in report['groups']]==[.5,.75,1.]
            assert all(len(g['trials'])==20 for g in report['groups'])
            checks.update(passed=True,discarded_updates=2,actor_updated=True,estimator_unchanged=True,
                          std_unchanged=True,standing_unchanged=True,checkpoint_roundtrip=True,evaluation_smoke=True,
                          evaluation_quality_passes=[g['passed'] for g in report['groups']])
            write(out/'checks.json',checks)
        status=json.loads((out/'status.json').read_text());status['state']='completed';write(out/'status.json',status)
        print('PREFLIGHT_PASSED' if opt.preflight else 'TRAINING_COMPLETE',out,flush=True)
    except BaseException as exc:
        status=json.loads((out/'status.json').read_text());status.update(state='failed',error=repr(exc));write(out/'status.json',status)
        raise
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)

if __name__=='__main__':main()
