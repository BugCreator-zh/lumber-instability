import torch 
import torch.nn as nn
from models.FPN import get_fpn

class Siamese_net(nn.Module):
    def __init__(self,config):
        super(Siamese_net,self).__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.Sim_measure = nn.Sequential( 
              nn.Linear(self.config.hidden_size, 512),  
              nn.LeakyReLU(),
              nn.Dropout(),
              nn.Linear(512, 128),
              nn.LeakyReLU(),
              nn.Dropout(),
              nn.Linear(128, 2),
              nn.Softmax(dim=1)
              )
        
    def forward(self,x):
        features1 = self.net(x[0])
        features2 = self.net(x[1])
        features1 = features1.flatten(1)
        features2 = features2.flatten(1)
        features = torch.abs(features1-features2)
        output = self.Sim_measure(features)
        return output
        
        
        