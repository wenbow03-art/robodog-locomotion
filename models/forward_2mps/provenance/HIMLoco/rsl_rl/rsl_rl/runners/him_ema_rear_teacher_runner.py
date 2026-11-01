"""Persist the moving teacher together with student/optimizers on every save."""
import torch
from .him_on_policy_runner import HIMOnPolicyRunner
from rsl_rl.modules.him_stance_width_actor import HIMStanceWidthActorCritic


class HIMEmaRearTeacherRunner(HIMOnPolicyRunner):
    def __init__(self,env,train_cfg,log_dir=None,device='cpu'):
        super().__init__(env,train_cfg,log_dir,device)
        from legged_gym.utils.lz_q1l_mirror import LZQ1LMirror
        teacher=HIMStanceWidthActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,
            env.num_actions,**self.policy_cfg).to(self.device)
        teacher.configure_right_rear(env.dof_names,env.commands_scale)
        self.alg.set_teacher(teacher,LZQ1LMirror.from_env(env,self.device),env.dof_names,env.commands_scale)
        self.reset_teacher_from_student()

    def reset_teacher_from_student(self):
        self.alg.teacher.load_state_dict(self.alg.actor_critic.state_dict(),strict=True)
        self.alg.teacher.requires_grad_(False).eval()
        self.alg.teacher_updates=0

    def initialize_from_checkpoint(self,path):
        info=super().load(path,load_optimizer=True)
        self.alg.learning_rate=self.alg.optimizer.param_groups[0]['lr']
        self.reset_teacher_from_student()
        return info

    def save(self,path,infos=None):
        torch.save(dict(model_state_dict=self.alg.actor_critic.state_dict(),
            optimizer_state_dict=self.alg.optimizer.state_dict(),
            estimator_optimizer_state_dict=self.alg.actor_critic.estimator.optimizer.state_dict(),
            teacher_state_dict=self.alg.teacher.state_dict(),teacher_updates=self.alg.teacher_updates,
            teacher_ema_tau=self.alg.teacher_ema_tau,iter=self.current_learning_iteration,infos=infos),path)

    def load(self,path,load_optimizer=True):
        saved=torch.load(path,map_location=self.device)
        if saved.get('teacher_ema_tau')!=self.alg.teacher_ema_tau or 'teacher_state_dict' not in saved:
            raise ValueError('EMA resume needs a saved teacher; use initialize_from_checkpoint for warm start')
        self.alg.actor_critic.load_state_dict(saved['model_state_dict'],strict=True)
        self.alg.teacher.load_state_dict(saved['teacher_state_dict'],strict=True)
        self.alg.teacher_updates=saved['teacher_updates']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
            self.alg.actor_critic.estimator.optimizer.load_state_dict(saved['estimator_optimizer_state_dict'])
            self.alg.learning_rate=self.alg.optimizer.param_groups[0]['lr']
        self.current_learning_iteration=saved['iter']
        return saved.get('infos')
