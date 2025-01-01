import torch.nn as nn
import torch
import math
class VGG(nn.Module):
    def __init__(self,num_classes=2,mode = 'test'):
        super(VGG,self).__init__()
        self.mode = mode
        self.embedding = torch.FloatTensor([])
        self.y = torch.IntTensor([])
        self.features = nn.Sequential(
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
        )
        self.last_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.last_pool = nn.Sequential(
                                       nn.ReLU(True),
                                       nn.MaxPool2d(kernel_size=2, stride=2)
                                       )
    def forward(self,x):
        x = self.features(x)
        x = self.last_conv(x)
        x = self.last_pool(x)
        x = x.view(x.size(0),-1)
            
        return x
