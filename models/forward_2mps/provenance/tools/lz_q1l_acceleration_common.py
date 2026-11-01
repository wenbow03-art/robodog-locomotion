"""Shared pinned checkpoints and standing-controller attachment for ramp tasks."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
TASK = 'lz_q1l_flat_forward_acceleration'
SOURCE = ROOT/'HIMLoco/legged_gym/logs2/flat_lz_q1l_forward_speed2/forward_speed2_300_20261030_224245_3985540/model_300.pt'
SOURCE_SHA = '0731696b5bc2ecd66fa0ec17449f9e76e84ddceadd0b8d6ad9d9ff31226f4def'
STANDING = ROOT/'HIMLoco/legged_gym/logs2/baselines/stand_walk800_20261027/gait_150.pt'
STANDING_SHA = '655a5022f2e1404c8f7c3dfbb670cc8235710f78e7550e72f2343e4e2c5e9747'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def attach_standing(env):
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils.helpers import class_to_dict
    from rsl_rl.modules import HIMActorCritic
    assert digest(STANDING) == STANDING_SHA
    _, train = task_registry.get_cfgs('lz_q1l_flat_v2_gait')
    model = HIMActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                         env.num_actions, **class_to_dict(train.policy)).to(env.device)
    model.load_state_dict(torch.load(STANDING, map_location=env.device)['model_state_dict'], strict=True)
    model.eval().requires_grad_(False)
    env.ramp_standing_model = model
    env.ramp_standing_policy = env.build_inference_policy(model)
    env.ramp_allow_bootstrap = False
    return model
