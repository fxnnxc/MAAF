import torch 
import torch.nn as nn 
from ..utils.make_backbone import make_backbone
from ..singleagent.adapter_ppo import MessageAdaptedAgent
from ..utils.message_handing import (make_handling_type, 
                                    message_feature_concat,
                                    compute_project_dim)

from .base import BaseMARL


class MAAF(BaseMARL):
    # generate messages and mean-pool the message representations 
    def __init__(self, network_infos:dict, network_mapping:dict, cooperative_mapping, origin_obs_dims:dict, origin_act_dims:dict,
                **kwargs):
        
        self.adapter_mode = False 
        self.current_handling_type = None
        self.message_handing_types = [[0,0,1]]+[
            make_handling_type(i) for i in range(1, 8)
        ]
        super().__init__(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, **kwargs)
        
    
    def step1(self, obs, feasible_actions=None):
        messages = [None for i in range(obs.size(1))]
        for i, j in self.network_mapping.items():
            a, p, _, probs, hiddens = self.shared_agents[j].get_action(obs[:,i,...], 
                                                        message=None,
                                                        action=None, 
                                                        feasible_action_dim=self.origin_act_dims[i],
                                                        feasible_actions=feasible_actions[i] if feasible_actions is not None else None,
                                                        )
            
            assert self.current_handling_type is not None, 'call set_message_handling_type first..'
            combined = message_feature_concat(
                                            self.current_handling_type,
                                            'discrete' if probs is not None else 'continuous',
                                            obs[:,i,...],
                                            a.detach().float() if a.ndim == hiddens.ndim else a.unsqueeze(1).detach().float(),
                                            probs.detach() if probs is not None else None,
                                            # hiddens.detach()
                                        )
            messages[i] = self.message_modules[str(self.current_handling_index) + "_" +str(j)](combined)
        message = torch.stack(messages, dim=1)
        return message
    
    def step2(self, obs, messages):
        pooled_messages = [None for k in self.cooperative_mapping.keys()]
        for k in self.cooperative_mapping.keys():
            # for each agent
            group_id = self.cooperative_mapping[k]
            gathered_messages = [] 
            for k2, other_group_id in self.cooperative_mapping.items():
                # for each agent
                if group_id == other_group_id:
                    gathered_messages.append(messages[:,k2, ...])
            pooled_messages[k] = self.pool(gathered_messages, group_id)
        
        pooled_messages = torch.stack(pooled_messages, dim=1)
        return pooled_messages
    

        
    def forward(self, obs, feasible_actions=None, return_message=False):
        if self.adapter_mode:
            message = self.step1(obs, feasible_actions)
            combined = self.step2(obs, message)
            if return_message:
                return combined, message
            else:
                return combined
        else:
            return None 
        
    def set_adapter_mode(self, condition, set_gradient_false=False):
        self.adapter_mode = condition
        for agent in self.shared_agents:
            agent.set_adapter_mode(condition, set_gradient_false)
            
    def set_message_handling_type(self, index):
        self.current_handling_index = index
        self.current_handling_type = self.message_handing_types[index]



class MAAFRNN(MAAF):
    # generate messages and mean-pool the message representations 
    def __init__(self, network_infos:dict, network_mapping:dict, cooperative_mapping, origin_obs_dims:dict, origin_act_dims:dict,
                **kwargs):
        super().__init__(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, **kwargs)
    
    def make_networks(self):
        # construction of agents 
        self.shared_agents = nn.ModuleList([
            MessageAdaptedAgent(**info) 
            for info in self.network_infos.values()
        ])
        
        # construction of message modules 
        message_modules = {}
        message_poolers = {}
        for i, info in self.network_infos.items():
            message_type = info['message_type']
            message_layers = info['message_layers']
            message_dim = info['message_dim']
            message_activation = info['message_activation']
            for index, message_handling_type in enumerate(self.message_handing_types):
                in_dim = compute_project_dim(
                    handing_type=message_handling_type, 
                    action_type=info['action_type'],  
                    probs_dim=info['action_dim'],
                    # hiddens_dim=info['hidden_dim'],
                    obs_dim=max(self.origin_obs_dims),
                    action_dim = 1 if info['action_type']=='discrete' else info['action_dim'], 
                    )
                module = make_backbone(f"{message_type}_{message_layers}_{message_dim}_{message_activation}", 
                                            in_dim,  
                                            proj_dim=message_dim, #add action to the message  
                                            no_final_act=False) 
                message_modules[str(index)+"_"+str(i)] = module
            message_poolers[str(i)] = nn.GRU(message_dim, message_dim, 2)
        self.message_poolers = nn.ModuleDict(message_poolers)
        self.message_modules = nn.ModuleDict(message_modules)

    def pool(self, gathered_messages, group_id):
        pooler = self.message_poolers[str(group_id)]
        x, hidden =pooler(torch.stack(gathered_messages, dim=1))
        x = x[:,-1,:]
        return x 


class MAAFATTN(MAAF):
    # generate messages and mean-pool the message representations 
    def __init__(self, network_infos:dict, network_mapping:dict, cooperative_mapping, origin_obs_dims:dict, origin_act_dims:dict,
                **kwargs):
        super().__init__(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, **kwargs)
    def make_networks(self):
        # construction of agents 
        self.shared_agents = nn.ModuleList([
            MessageAdaptedAgent(**info) 
            for info in self.network_infos.values()
        ])
        
        # construction of message modules 
        message_modules = {}
        message_poolers = {}
        for i, info in self.network_infos.items():
            message_type = info['message_type']
            message_layers = info['message_layers']
            message_dim = info['message_dim']
            assert message_dim % 3 == 0
            message_activation = info['message_activation']
            for index, message_handling_type in enumerate(self.message_handing_types):
                in_dim = compute_project_dim(
                    handing_type=message_handling_type, 
                    action_type=info['action_type'],  
                    probs_dim=info['action_dim'],
                    # hiddens_dim=info['hidden_dim'],
                    obs_dim=max(self.origin_obs_dims),
                    action_dim = 1 if info['action_type']=='discrete' else info['action_dim'], 
                    )
                module = make_backbone(f"{message_type}_{message_layers}_{message_dim}_{message_activation}", 
                                            in_dim,  
                                            proj_dim=message_dim*3, #add action to the message  
                                            no_final_act=False) 
                message_modules[str(index)+"_"+str(i)] = module
            message_poolers[str(i)] = nn.MultiheadAttention(message_dim, 1, batch_first=True)
            
        self.message_poolers = nn.ModuleDict(message_poolers)
        self.message_modules = nn.ModuleDict(message_modules)

    def pool(self, gathered_messages, group_id):
        gathered_messages = torch.stack(gathered_messages, dim=1)
        hidden_dim = gathered_messages.shape[-1]//3
        query, key, value = gathered_messages.split([hidden_dim, hidden_dim, hidden_dim], dim=-1)
        pooler = self.message_poolers[str(group_id)]
        x, attn_output_weights = pooler(query, key, value)
        x = x.mean(dim=1)
        return x 
