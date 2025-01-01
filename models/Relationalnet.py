import torch
import torch.nn as nn
from .multiple_input_net import MIN


class SM(nn.Module):
    def __init__(self, config,model_name):
        super(SM,self).__init__()
        self.config = config
        self.Min = MIN(config,model_name).to(config.device)
        self.hidden_state = None
        self.last_state = None
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        hidden_sizes = {'resnet18':1280,'googlenet':131072,'alexnet':0}
        self.hidden_size = hidden_sizes[model_name[0]]

        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size, 128),
                        nn.BatchNorm1d(128),
                        nn.LeakyReLU(),
                        nn.Linear(128, 2),
                        nn.Softmax(dim = 1)
            )

    def distance(self,v1,v2):
        return torch.abs(v1-v2)
        
    def forward(self, data1,data2):
        result1 = self.Min(data1[0],data1[1])
        result2 = self.Min(data2[0],data2[1])
        #距离计算
        result = self.distance(result1,result2)

        self.hidden_state = result
        result = self.classify(result)
        return result
