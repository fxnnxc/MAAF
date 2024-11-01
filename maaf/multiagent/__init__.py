
from .mappo import MAPPO
from .maaf import MAAFRNN, MAAFATTN
from .tarmac import TarMAC
from .ippo import IPPO


def get_agent(name, network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims,  **kwargs):
    if name == "maaf_rnn":
        return MAAFRNN(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, )
    if name == "maaf_attn":
        return MAAFATTN(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, )
    if name == "mappo":
        return MAPPO(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, )
    if name == "ippo":
        return IPPO(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, )
    if name == "tarmac":
        return TarMAC(network_infos, network_mapping, cooperative_mapping, origin_obs_dims, origin_act_dims, )
    raise ValueError("not implemented")

def make_network_info(agent_names, max_obs_dim, max_act_dim,
                      message_hidden_dim,message_dim,message_layers,message_activation,
                      num_layers=4, hidden_dim=64, activation='ReLU', proj_dim=64, action_type='discrete'):
    print(agent_names)
    print(agent_names)
    
    name_to_idx = {}
    network_infos = {}
    network_mapping = {}
    cooperative_mapping= {}
    for i, name in enumerate(agent_names):
        agent_type, index = name.split("_")
        if agent_type not in name_to_idx:
            name_to_idx[agent_type] = len(name_to_idx)
        network_infos[i] = dict(
            backbone_type='mlp', 
            num_layers=num_layers, 
            hidden_dim=hidden_dim, 
            activation=activation, 
            in_features=max_obs_dim, 
            proj_dim=proj_dim, 
            action_dim=max_act_dim, 
            action_type=action_type, 
            no_final_act=False,
            message_type='mlp',
            message_activation=message_activation,
            message_layers=message_layers,
            message_dim=message_dim,
            message_hidden_dim=message_hidden_dim,
        )
        network_mapping[i] = i
        cooperative_mapping[i] = name_to_idx[agent_type] # 0 # all the agents share information /  name_to_idx[agent_type]
    return network_infos, network_mapping, cooperative_mapping

