"""V4 inference: unchanged stateful baseline outside pure lateral commands."""
import torch
from .lz_q1l_stand_walk import StandWalkResidualActor


class LateralResidualActor(StandWalkResidualActor):
    policy_mode='v4-new-baseline+direct-lateral-no-mirror'

    def __call__(self,history):
        return self.act_inference(history)

    def mirror_mask(self,history):
        return torch.zeros(len(history),dtype=torch.bool,device=history.device)

    @torch.no_grad()
    def act_inference(self,history):
        baseline=super().act_inference(history)
        mask=self.model.lateral_mask(history)
        if not mask.any():return baseline
        result=baseline.clone()
        result[mask]=self.model.lateral_actions(history[mask])
        return result
