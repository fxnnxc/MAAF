import torch 
import torch.nn as nn 
from .base import CommonPPOModule
from ..utils.make_backbone import make_backbone, layer_init

def make_controller(proj_dim ,action_dim,  std=1.0):
    controller = layer_init(nn.Linear(proj_dim, action_dim), std=std)
    return controller

class PPOAgent(CommonPPOModule):
    def __init__(self, backbone_type, 
                 num_layers, 
                 hidden_dim, 
                 activation, 
                 in_features, 
                 proj_dim, 
                 action_dim, 
                 action_type, 
                 no_final_act=False):
        super().__init__()
        backbone = f"{backbone_type}_{num_layers}_{hidden_dim}_{activation}"
        self.action_type = action_type
        self.backbone_critic = make_backbone(backbone, in_features, proj_dim, no_final_act)
        self.backbone_actor  = make_backbone(backbone, in_features, proj_dim, no_final_act)
        self.critic = layer_init(nn.Linear(proj_dim, 1))
        if self.action_type == "discrete":
            self.controller = make_controller(proj_dim ,action_dim, std=1.0)
        else:
            self.controller = make_controller(proj_dim ,action_dim, std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        x = self.backbone_critic(x)
        return self.critic(x)

    def get_action(self, x, action=None, feasible_action_dim=None, deterministic=False, feasible_actions=None):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.backbone_actor(x)
        if self.action_type == "continuous":
            action_mean = self.controller(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            action, logprob, entropy, pp = self.compute_discrete_action(action, action_std)
        else:
            logits = self.controller(x)
            action, logprob, entropy, pp = self.compute_discrete_action(action, logits, feasible_action_dim, deterministic, feasible_actions)
            
        return action, logprob, entropy, pp, x 
