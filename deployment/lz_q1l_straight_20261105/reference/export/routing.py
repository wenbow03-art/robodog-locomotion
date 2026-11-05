import torch
# ids 1,2,3 are residuals. 0 means standing/base-only/high-speed.
def route(cmd, history):
    ids=torch.where((cmd>0)&(cmd<=1.5),1+(cmd>.5).long()+(cmd>1.).long(),torch.zeros_like(cmd,dtype=torch.long))
    hist=torch.cat((history,ids[:,None]),1)
    hist=torch.where((cmd>0)[:,None],hist,torch.ones_like(hist))
    weights=torch.stack([(hist==i).float().mean(1) for i in (1,2,3)],dim=1)
    return weights,hist[:,1:].contiguous()

def bounds(idx):
    return (.5*idx,.5*(idx+1))
