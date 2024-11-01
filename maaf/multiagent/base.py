import torch
import torch.nn as nn 
from ..utils.make_backbone import make_backbone

class BaseMARL(nn.Module):
    def __init__(self, 
                 network_infos:dict, 
                 network_mapping:dict, 
                 cooperative_mapping:dict,
                 origin_obs_dims:dict, 
                 origin_act_dims:dict, 
                 print_state=True, *args, **kwargs):
        super().__init__()
        self.network_infos = network_infos        
        self.network_mapping = network_mapping
        self.cooperative_mapping = cooperative_mapping
        self.origin_obs_dims = origin_obs_dims
        self.origin_act_dims = origin_act_dims
        
        for k, v in network_infos.items():
            v['in_features'] = max(origin_obs_dims) # + msg_dim
        
        if print_state:
            print("----------------")
            print(network_infos)
            print(network_mapping)
            print("----------------")
        
        for k, v in self.network_mapping.items():
            assert v in self.network_infos, f"'{v}' must be in network_infos"
        self.make_networks()
        
    def get_message(self, x):
        return None 
    
    def make_networks(self):
        raise NotImplementedError()

    def get_value(self, x):
        # x: batch x agents x dims 
        messages = None  # We restrict the contribution of messages to the Values    #  self.get_message(x)
        v = [] 
        for i, mapping in self.network_mapping.items():
            agent = self.shared_agents[mapping]
            m = None if messages is None else messages[:,i,...]
            v.append(agent.get_value(x[:,i,:], m))
        v = torch.stack(v, dim=1)
        return v

    def get_action(self, x, actions=None, deterministic=False, feasible_actions=None):
        # batch x agents x dims 
        messages = self.forward(x, feasible_actions)
        
        a = [] 
        p = []
        e = [] 
        for i, mapping in self.network_mapping.items():
            agent = self.shared_agents[mapping]
            if actions is not None:
                act = actions[:,i]
            else:
                act = None 
            m = None if messages is None else messages[:,i,...]
            action, logprob, entropy, probs, hiddens = agent.get_action(x[:,i,...], 
                                                        m,
                                                        action=act, 
                                                        feasible_action_dim=self.origin_act_dims[i],
                                                        deterministic=deterministic,
                                                        feasible_actions=feasible_actions[i] if feasible_actions is not None else None)
            a.append(action)
            p.append(logprob)
            e.append(entropy)
        a = torch.stack(a, dim=1)
        p = torch.stack(p, dim=1)
        e = torch.stack(e, dim=1)
        return a, p, e

    def get_action_and_value(self, x, action=None, deterministic=False, feasible_actions=None):
        action, logprob, entropy = self.get_action(x, action, deterministic, feasible_actions=feasible_actions)
        value = self.get_value(x)
        return action, logprob, entropy, value
