import torch 
import torch.nn as nn 
import numpy as np 
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from ..utils.make_backbone import make_backbone, layer_init

def make_controller(proj_dim ,action_dim,  std=1.0):
    controller = layer_init(nn.Linear(proj_dim, action_dim), std=std)
    return controller

class CommonPPOModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def compute_continuous_action(self, action, action_mean, action_std, deterministic):
        probs = Normal(action_mean, action_std)
        if action is None:
            if deterministic:
                raise NotImplementedError()
            else:
                action = probs.sample()
        # print(action.size(), probs.log_prob(action).size(), probs.entropy().size())
        if hasattr(probs, 'probs'):
            pp = probs.probs 
        else:
            pp = None
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), pp
        
    def compute_discrete_action(self, action, logits, feasible_action_dim, deterministic, feasible_actions):    
        logits[:,feasible_action_dim:] = -np.inf 
        if feasible_actions is not None:
            for j, k in enumerate(feasible_actions):
                if k ==0:
                    logits[:, j] = -np.inf
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = logits.argmax(dim=1)
            else:
                action = probs.sample()
        if hasattr(probs, 'probs'):
            pp = probs.probs 
        else:
            pp = None
        return action, probs.log_prob(action), probs.entropy(), pp 
