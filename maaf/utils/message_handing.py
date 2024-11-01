import torch 


# [1,0,0,0]
# message module layers 0, 1, 2   
def compute_project_dim(handing_type, action_type, obs_dim, action_dim, probs_dim, ):
    dim = 0
    if handing_type[0] == 1:
        dim += obs_dim
    if handing_type[1] == 1:
        dim += action_dim
    if handing_type[2] == 1 and action_type =='discrete':
        dim += probs_dim
    # if handing_type[3] == 1:
    #     dim += hiddens_dim
    return dim 

def message_feature_concat(handing_type, action_type, obs, action, logits, ):
    v = []
    if handing_type[0] == 1:
        v.append(obs)
    if handing_type[1] == 1:
        v.append(action)
    if handing_type[2] == 1 and action_type == 'discrete':
        v.append(logits)
    # if handing_type[3] == 1:
    #     v.append(hiddens)
    return torch.concat(v, dim=-1)




def make_handling_type(handling_index):
    assert 1<=handling_index<8
    v = [0,0,0]
    b = bin(handling_index)[2:]
    b = ''.join(['0']*(3-len(b))) + str(b)
    for i, k in enumerate(b):
        if k == "1":
            v[i] = 1
    return v 
