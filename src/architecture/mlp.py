import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch
import torch.nn as nn
from conf.config import MyConfig
import torch.nn.init as init
import math

def get_model(cfg:MyConfig=None):
    num_classes = cfg.model.output_size #number of classes 
    input_size = int(cfg.model.input_size)
    model = MLP(num_classes, input_size, cfg)

    path = cfg.model.pretrained_path
    if path is not None:
        weights=torch.load(path, weights_only=True)
        model.load_state_dict(weights)
    else:
        model.apply(init_parameters)
    return model

def init_parameters(module):
    if isinstance(module, nn.Linear):
        init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            module.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, num_classes, input_size, cfg:MyConfig):
        super(MLP, self).__init__()

        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.squeeze(-1)
        return x