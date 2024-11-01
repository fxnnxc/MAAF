import torch 
import torch.nn as nn 
from .base import CommonPPOModule
from ..utils.make_backbone import make_backbone, layer_init

def make_controller(proj_dim ,action_dim,  std=1.0):
    controller = layer_init(nn.Linear(proj_dim, action_dim), std=std)
    return controller

class MessageAdaptedAgent(CommonPPOModule):
    def __init__(self, backbone_type, 
                 num_layers, 
                 hidden_dim, 
                 activation, 
                 in_features, 
                 message_dim,
                 proj_dim, 
                 action_dim, 
                 action_type, 
                 no_final_act=False,
                 **kwargs):
        super().__init__()
        backbone = f"{backbone_type}_{num_layers}_{hidden_dim}_{activation}"
        self.action_type = action_type
        self.base_backbone_critic = make_backbone(backbone, in_features, proj_dim, no_final_act)
        self.base_backbone_actor  = make_backbone(backbone, in_features, proj_dim, no_final_act)
        self.adapter_backbone_critic = make_backbone(backbone, message_dim, proj_dim, no_final_act)
        self.adapter_backbone_actor  = make_backbone(backbone, message_dim, proj_dim, no_final_act)
    
        self.base_critic = layer_init(nn.Linear(proj_dim, 1))
        self.adapter_critic = layer_init(nn.Linear(proj_dim, 1))
        if self.action_type == "discrete":
            self.base_controller = make_controller(proj_dim ,action_dim, std=1.0)
            self.adapter_controller = make_controller(proj_dim ,action_dim, std=1.0)
        else:
            self.base_controller = make_controller(proj_dim ,action_dim, std=0.01)
            self.base_actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            
            self.adapter_controller = make_controller(proj_dim ,action_dim, std=0.01)
            self.adapter_actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            
        self.adapter_mode = False 

    def set_adapter_mode(self, condition, set_gradient_false=False):
        if condition:
            self.base_controller.requires_grad_(True)
            self.base_critic.requires_grad_(True)
            if self.action_type != 'discrete':
                self.base_actor_logstd.requires_grad_(True)
            self.base_backbone_critic.requires_grad_(True)
            self.base_backbone_actor.requires_grad_(True)
            if set_gradient_false:
                self.base_controller.requires_grad_(False)
                self.base_critic.requires_grad_(False)
                if self.action_type != 'discrete':
                    self.base_actor_logstd.requires_grad_(False)
                self.base_backbone_critic.requires_grad_(False)
                self.base_backbone_actor.requires_grad_(False)
            self.adapter_mode = True 
        else:
            self.base_controller.requires_grad_(True)
            self.base_critic.requires_grad_(True)
            if self.action_type != 'discrete':
                self.base_actor_logstd.requires_grad_(True)
            self.base_backbone_critic.requires_grad_(True)
            self.base_backbone_actor.requires_grad_(True)
            self.adapter_mode = False 

    def get_value(self, x, message=None,):
        if self.adapter_mode and message is not None:
            x = self.base_backbone_critic(x)
            x = self.base_critic(x)
            y = self.adapter_backbone_critic(message)
            y = self.adapter_critic(y)
            x = x + y
        else:
            x = self.base_backbone_critic(x)
            x = self.base_critic(x)
        return x 

    def get_action(self, x, message=None, action=None, feasible_action_dim=None, deterministic=False, feasible_actions=None):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = self.base_backbone_actor(x)
        if self.adapter_mode and message is not None :
            y = self.adapter_backbone_actor(message)
            
        if self.action_type == "continuous":
            if not self.adapter_mode or message is None:                
                action_mean = self.base_controller(x)
                action_logstd = self.base_actor_logstd.expand_as(action_mean, action_std, deterministic)
            else:
                base_action_mean = self.base_controller(x)
                adapter_action_mean = self.adapter_controller(y)
                action_mean = base_action_mean + adapter_action_mean
                action_logstd = self.adapter_actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            action, logprob, entropy, pp = self.compute_discrete_action(action, action_std)
        else:
            if not self.adapter_mode or message is None:      
                logits = self.base_controller(x)          
            else:
                base_logits = self.base_controller(x)          
                adapter_logits = self.adapter_controller(y)          
                logits = base_logits + adapter_logits
            
            action, logprob, entropy, pp = self.compute_discrete_action(action, logits, feasible_action_dim, deterministic, feasible_actions)
        return action, logprob, entropy, pp, x 
            