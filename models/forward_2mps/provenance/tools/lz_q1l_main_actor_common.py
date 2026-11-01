"""Pinned source for the user-authorized direct removal of RH correction."""
from lz_q1l_acceleration_common import ROOT, STANDING, STANDING_SHA, attach_standing, digest, write

TASK = 'lz_q1l_flat_forward_main_actor'
SOURCE = ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_forward_acceleration/acceleration_500_20261031_005110_4024454/model_300.pt'
SOURCE_SHA = 'ffc4274b9d81e38485878ebefe4bd4a71c522274ab78877f201ad30f63d5fec7'
FILES = [
    'tools/lz_q1l_main_actor_common.py', 'tools/train_lz_q1l_forward_main_actor.py',
    'tools/evaluate_lz_q1l_forward_main_actor.py', 'tools/launch_lz_q1l_forward_main_actor.py',
    'tools/run_lz_q1l_forward_main_actor.sh',
    'HIMLoco/rsl_rl/rsl_rl/modules/him_main_actor.py',
    'HIMLoco/rsl_rl/rsl_rl/algorithms/him_main_actor_ppo.py',
]


def load_main_actor(runner, env, saved, old_policy_cfg):
    """Copy retained weights exactly, remap retained Adam state by parameter name."""
    import copy
    import torch
    from rsl_rl.modules.him_stance_width_actor import HIMStanceWidthActorCritic
    from train_lz_q1l_stance_width import same
    model = runner.alg.actor_critic
    old = HIMStanceWidthActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                                  env.num_actions, **old_policy_cfg).to(env.device)
    old.configure_right_rear(env.dof_names, env.commands_scale)
    old.load_state_dict(saved['model_state_dict'], strict=True)
    old.eval()
    removed = set(saved['model_state_dict']) - set(model.state_dict())
    expected = {k for k in saved['model_state_dict'] if k.startswith('right_rear_delta.')}
    expected |= {'right_rear_indices', 'command_scale'}
    assert removed == expected
    retained = {k:saved['model_state_dict'][k] for k in model.state_dict()}
    model.load_state_dict(retained, strict=True)
    assert same(retained, model.state_dict())

    old_names = {id(p):name for name,p in old.named_parameters()}
    old_order = [old_names[id(p)] for p in old.optimization_parameters()]
    old_optimizer = saved['optimizer_state_dict']
    assert len(old_optimizer['param_groups']) == 1
    old_ids = old_optimizer['param_groups'][0]['params']
    assert len(old_ids) == len(old_order)
    by_name = dict(zip(old_order, old_ids))
    new_names = {id(p):name for name,p in model.named_parameters()}
    new_order = [new_names[id(p)] for p in model.optimization_parameters()]
    new_ids = runner.alg.optimizer.state_dict()['param_groups'][0]['params']
    states = {}
    for name, idx in zip(new_order, new_ids):
        prior = old_optimizer['state'][by_name[name]]
        param = dict(model.named_parameters())[name]
        assert prior['exp_avg'].shape == param.shape == prior['exp_avg_sq'].shape
        states[idx] = copy.deepcopy(prior)
    group = copy.deepcopy(old_optimizer['param_groups'][0]);group['params'] = new_ids
    group['lr'] = runner.alg.learning_rate
    runner.alg.optimizer.load_state_dict(dict(state=states,param_groups=[group]))
    restored = runner.alg.optimizer.state_dict()
    assert all(same(states[idx], restored['state'][idx]) for idx in new_ids)
    model.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
    assert same(model.estimator.optimizer.state_dict(),saved['estimator_optimizer_state_dict'])

    # Same observations, exact main outputs; the other nine columns equal old policy.
    with torch.no_grad():
        obs = env.get_observations().clone()
        obs[:,0] = torch.linspace(0.,.4,len(obs),device=env.device)*env.commands_scale[0]
        vel,latent=old.estimator(obs)
        reference=old.actor(torch.cat((obs[:,:env.num_one_step_obs],vel,latent),-1))
        actual=model.act_inference(obs)
        assert torch.equal(reference,actual)
        model.update_distribution(obs)
        assert torch.equal(model.action_mean,actual)
        other=[i for i in range(12) if i not in old.right_rear_indices.tolist()]
        assert torch.equal(old.act_inference(obs)[:,other],actual[:,other])
    assert not hasattr(model,'right_rear_delta') and not hasattr(model,'correction_mask')
    assert all(p.requires_grad for p in model.actor.parameters())
    assert all(not p.requires_grad for p in model.estimator.parameters()) and not model.std.requires_grad
    return dict(exact_retained_weights=True,removed_state_keys=sorted(removed),
                optimizer_moments_restored_by_name=new_order,exact_estimator_optimizer=True,
                training_and_inference_equal_main_actor=True,other_nine_outputs_unchanged_on_same_obs=True,
                source_iter=int(saved['iter']),learning_rate=runner.alg.learning_rate,
                note='Direct branch removal, no distillation, no output folding; EMA/RH imitation discarded.')
