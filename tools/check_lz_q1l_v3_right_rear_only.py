"""Frozen replay and backward-only checks; no optimizer step and no training."""
import hashlib
import io
import json
import sys
from unittest.mock import patch

import isaacgym
import torch
from legged_gym.envs import task_registry
from legged_gym.utils import get_args
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.runners import HIMOnPolicyRunner
from rsl_rl.modules.him_right_rear_only import HIMRightRearOnlyActorCritic
from train_lz_q1l_v3_right_rear_only import (
    ROOT, REVIEW, SOURCE, SOURCE_SHA256, TASK, NEW_SOURCES,
    verify_baseline, initialize_student, attach_teacher)


def state(module):return {k:v.detach().clone() for k,v in module.state_dict().items()}
def unchanged(module,before):return all(torch.equal(v,before[k]) for k,v in module.state_dict().items())


def main():
    verify_baseline();REVIEW.mkdir(parents=True,exist_ok=False)
    cfg,train=task_registry.get_cfgs(TASK)
    archived=json.loads((SOURCE.parent/'env_config.json').read_text())
    archived['domain_rand'].pop('push_interval',None)
    assert class_to_dict(cfg)==archived
    assert train.runner.max_iterations==0 and train.algorithm.other_legs_coef==0
    cfg.env.num_envs=6;cfg.env.episode_length_s=20.;cfg.commands.resampling_time=1e6
    cfg.noise.add_noise=False;cfg.sim.use_gpu_pipeline=cfg.sim.physx.use_gpu=False;cfg.sim.physx.num_threads=2
    for key in dir(cfg.domain_rand):
        if not key.startswith('_') and isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    sys.argv=[sys.argv[0],'--task',TASK,'--headless','--seed','1','--sim_device','cpu','--rl_device','cpu','--num_envs','6']
    env=None
    try:
        env,_=task_registry.make_env(TASK,args=get_args(),env_cfg=cfg)
        runner=HIMOnPolicyRunner(env,class_to_dict(train),None,device='cpu')
        initialize_student(runner);attach_teacher(runner,env,train)
        alg=runner.alg;student=alg.actor_critic;teacher=alg.teacher
        before=state(student);teacher_before=state(teacher)
        rh=student.right_rear_indices;other=alg.other_indices
        assert [env.dof_names[i] for i in rh.tolist()]==['roll_leg_b_r_joint','pitch01_leg_b_r_joint','pitch02_leg_b_r_joint']
        obs=torch.randn(24,270);obs[:,:3]=0.;obs[12:,0]=.35*env.commands_scale[0]
        with torch.no_grad():
            expected=teacher.act_inference(obs)
            assert torch.equal(student.act_inference(obs),expected)
        # Deliberately perturb the temporary, in-memory RH correction to prove
        # architecture isolation, then restore it. This is not an optimizer update.
        branch=state(student.right_rear_delta)
        with torch.no_grad():
            student.right_rear_delta[-1].weight.normal_(0,.1)
            student.right_rear_delta[-1].bias.copy_(torch.tensor([.3,-.4,.5]))
            moved=student.act_inference(obs)
            assert torch.equal(moved[:,other],expected[:,other])
            assert not torch.equal(moved[:,rh],expected[:,rh])
            for command in [[0,.3,0],[0,0,.4],[-.3,0,0]]:
                outside=obs.clone();outside[:,:3]=torch.tensor(command)*env.commands_scale
                assert torch.equal(student.act_inference(outside),teacher.act_inference(outside))
            student.right_rear_delta.load_state_dict(branch)
        assert unchanged(student,before)
        student.train();assert not student.actor.training and not student.estimator.training
        student.update_distribution(obs)
        assert torch.equal(student.action_std,teacher.std.expand_as(student.action_std))
        target,_=alg.teacher_targets(obs)
        left=[env.dof_names.index(k+'_leg_b_l_joint') for k in ('roll','pitch01','pitch02')]
        with torch.no_grad():
            left_action=teacher.act_inference(alg.mirror.history(obs))[:,left]*torch.tensor([-1.,1.,1.])
        assert torch.equal(target,left_action)
        loss,stats=alg.distillation_loss(obs,student.action_mean)
        alg.optimizer.zero_grad();loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.right_rear_delta.parameters())
        assert all(p.grad is None for p in student.frozen_policy_parameters())
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        alg.assert_frozen_policy()
        assert not alg.optimizer.state and not student.estimator.optimizer.state
        checks=dict(environment_exactly_baseline=True,named_RH_only=True,
            zero_branch_equals_baseline_all12=True,arbitrary_RH_branch_other9_bitwise_identical=True,
            outside_stand_forward_baseline_all12=True,actor_estimator_std_frozen=True,
            frozen_parameters_excluded_from_optimizer=True,LH_teacher_to_RH_only=True,
            supervision_gradients_only_RH_branch=True,teacher_frozen=True,fresh_optimizer_no_old_momentum=True)
        policy=runner.get_inference_policy(device='cpu');obs,_=env.reset();resets=[]
        with torch.no_grad():
            for step in range(250):
                env.commands.zero_();env.command_sample_mode.zero_()
                if step>=100:env.commands[3:,0]=.35;env.command_sample_mode[3:]=1
                obs[:,:3]=env.commands[:,:3]*env.commands_scale
                actions=policy(obs)
                assert torch.equal(actions,teacher.act_inference(obs))
                obs,_,reward,done,_,_,_=env.step(actions)
                assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                resets.extend(dict(step=step,index=i) for i in done.nonzero().flatten().tolist())
        checks['250_step_frozen_replay_exactly_baseline']=True
        alg.init_storage(6,4,[env.num_obs],[env.num_privileged_obs],[12])
        with torch.no_grad():
            for _ in range(4):
                critic=env.get_privileged_observations()
                actions=alg.act(obs.clone(),critic.clone())
                obs,next_critic,reward,done,info,_,_=env.step(actions)
                alg.process_env_step(reward,done,info,next_critic)
            alg.compute_returns(next_critic)
        alg.num_learning_epochs=alg.num_mini_batches=1
        with patch.object(student.estimator,'update',side_effect=AssertionError('Frozen estimator update called')) as forbidden:
            with patch.object(alg.optimizer,'step',return_value=None) as intercepted:
                losses=alg.update()
                assert intercepted.call_count==1 and forbidden.call_count==0
        assert all(torch.isfinite(torch.tensor(v)) for v in losses)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.critic.parameters())
        assert all(p.grad is None for p in student.frozen_policy_parameters())
        assert unchanged(student,before) and unchanged(teacher,teacher_before)
        checks['PPO_backward_only_estimator_never_updated']=True
        checks['all_model_parameters_unchanged']=True
        # The artifact includes the frozen baseline + RH branch; ordinary strict
        # load retains isolation and inference works without a teacher object.
        buffer=io.BytesIO();torch.save(student.state_dict(),buffer);buffer.seek(0)
        restored=HIMRightRearOnlyActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,12,
                                           **class_to_dict(train.policy))
        restored.configure_right_rear(env.dof_names,env.commands_scale)
        restored.load_state_dict(torch.load(buffer),strict=True)
        with torch.no_grad():assert torch.equal(restored.act_inference(obs),student.act_inference(obs))
        checks['strict_checkpoint_roundtrip_without_teacher']=True
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA256
        sources=set(json.loads((SOURCE.parent/'run_metadata.json').read_text())['code_sha256'])|set(NEW_SOURCES)
        report=dict(checks=checks,training_updates=0,parameters_unchanged=True,checkpoint_sha256=SOURCE_SHA256,
            replay=dict(envs=6,steps=250,resets=resets),backward_only=dict(losses=losses,optimizer_step_intercepted=True),
            trainable_parameter_groups=['right_rear_delta','critic'],frozen_parameter_groups=['actor','estimator','std'],
            temporary_test_branch_perturbation_restored=True,
            source_sha256={str(ROOT/p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(sources)})
        (REVIEW/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('RIGHT_REAR_ONLY_CHECKS_OK',json.dumps(dict(checks=checks,resets=resets,training_updates=0)),flush=True)
    finally:
        if env is not None:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
