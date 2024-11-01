
import torch 
import torch.nn as nn 
import numpy as np 

class BackboneBase(nn.Module):
    def __init__(self, backbone_type):
        super().__init__()
        self.backbone_type = backbone_type
    
    def get_layers_index(self, module_or_act):
        assert module_or_act in ['module', 'act'] , module_or_act
        layers_index = []
        for i, module in enumerate(self.backbone):
            if self.backbone_type == "mlp":
                if module.__class__.__name__ == "Linear":
                    layers_index.append(i)
            elif self.backbone_type == "cnn":
                if module.__class__.__name__ == "Conv2d":
                    layers_index.append(i)
        if module_or_act == "act":
            layers_index = [i+1 for i in layers_index]
        return layers_index

class MLPBackbone(BackboneBase):
    def __init__(self, num_layers, observation_dim, hidden_dim, proj_dim, activation, no_final_act):
        super().__init__('mlp')
        net = [
            layer_init(nn.Linear(observation_dim, hidden_dim)),
            activation(),
        ]
        for i in range(num_layers-1):
            if i == num_layers-2:
                net.append(layer_init(nn.Linear(hidden_dim, proj_dim))),
            else:
                net.append(layer_init(nn.Linear(hidden_dim, hidden_dim))),
            if not(i == num_layers-2 and no_final_act):
                net.append(activation())
        self.backbone = nn.Sequential(*net)
        
    def forward(self, x):
        return self.backbone(x)

    def get_activations(self):
        return super().get_activations()
    def get_layer_names(self):
        return super().get_layer_names()

class CNNBackbone(BackboneBase):
    def __init__(self, in_channels, proj_dim, activation, no_final_act):
        super().__init__('cnn')
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            activation(),
            nn.Conv2d(32, 64, 4, stride=2),
            activation(),
            nn.Conv2d(64, 64, 3, stride=1),
            activation(),
            nn.Flatten(),
            nn.Linear(3136, proj_dim),
            activation(),
        )
    def forward(self, x):
        return self.backbone(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_backbone(backbone, observation_size, proj_dim, no_final_act=False ):
    backbone_info = backbone 
    observation_dim = np.prod(observation_size)
    
    module = backbone_info.split("_")[0]
    if module == "mlp":
        _, num_layers, hidden_dim, activation = backbone_info.split("_")
        num_layers = int(num_layers)
        hidden_dim = int(hidden_dim)
        activation = getattr(nn, activation)
        if num_layers ==1:
            hidden_dim = proj_dim
        backbone = MLPBackbone(num_layers, observation_dim, hidden_dim, proj_dim, activation, no_final_act)
        
    elif "cnn_atari" in  backbone :
        activation = getattr(nn, backbone.split("_")[-1])
        backbone = CNNBackbone(in_channels, proj_dim, activation, no_final_act)
    else:
        raise ValueError()    
    return backbone

if __name__ == "__main__":
    model = MLPBackbone(4, 6, 8, 4, nn.ReLU, no_final_act=False)
    print(model.get_layers_index('module'))
    print(model.get_layers_index('act'))
    print([model.backbone[i] for i in model.get_layers_index("act")])
    model = CNNBackbone(3, 64, nn.ReLU, no_final_act=False)
    print(model.get_layers_index('module'))
    print(model.get_layers_index('act'))
    print([model.backbone[i] for i in model.get_layers_index("act")])
    
    model = make_backbone("mlp_2_32_ReLU", 4, 64, no_final_act=True)
    print(model)