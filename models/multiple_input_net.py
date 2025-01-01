import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from .resnet import resnet18
from .Alexnet import AlexNet
from .Googlenet import GoogLeNet
from .vggnet import VGG


class MIN(nn.Module):
    def __init__(self, config,model_name):
        super(MIN,self).__init__()
        self.config = config
        self.model_list = {'resnet18':resnet18(),'googlenet':GoogLeNet(),'alexnet':AlexNet(),'vggnet':VGG()}
        self.sag_model = self.model_list[model_name[0]]
        self.tra_model = self.model_list[model_name[1]]
        self.hidden_state = None
        self.last_state = None
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        hidden_sizes = {'resnet18':1024,'googlenet':131072,'alexnet':18432,'vggnet':65536} #normal:65536
        self.hidden_size = hidden_sizes[model_name[0]]
        #self.encoder = nn.LSTM(input_size=32768,hidden_size=256,num_layers=5,bidirectional=False)
    
        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size, 256),
                        nn.BatchNorm1d(256),
                        nn.LeakyReLU(),
                        nn.Dropout(p = 0.5),
                        nn.Linear(256, 3), #三分类时将2修改为3
                        nn.Softmax(dim = 1)
            )
 
        """
        self.classify = nn.Sequential(
                        nn.Linear(self.hidden_size ,512),
                        nn.LeakyReLU(),
                        nn.Dropout(),
                        nn.Linear(512, 128),
                        nn.LeakyReLU(),
                        nn.Dropout(),
                        nn.Linear(128, 2),
                        nn.Softmax(dim=1)
                                     )
        """
        
    def forward(self, x1,x2 = None):
        result = torch.FloatTensor([]).to(self.config.device)
        #embed = torch.FloatTensor([]).to(self.config.device)
        for i in range(len(x1)):
            #features = torch.FloatTensor([]).to(self.config.device)
            #embeddings = torch.FloatTensor([]).to(self.config.device)
            features = self.tra_model(x2[i].unsqueeze(0))
            for j in range(len(x1[0])):
                feature = self.sag_model(x1[i][j].unsqueeze(0)) + j/len(x1[0])
                features = torch.cat([features,feature])
                #embeddings = torch.cat([embeddings,feature])
            features = features.flatten(0).unsqueeze(0)
            result = torch.cat([result,features])
            #embed = torch.cat([embed,embeddings.unsqueeze(0)])
        """
        #BILSTM
        embed = embed.transpose(0,1)
        outputs, _ = self.encoder(embed)  # output, (h, c)
        encoding = outputs[-1]
        #encoding = torch.cat((outputs[0], outputs[-1]), -1)
        self.hidden_state = encoding
        result = self.classify(encoding)
        return result
        """
        #normal
        result = self.max_pool(result.unsqueeze(0))[0]
        self.hidden_state = result
        result = self.classify(result)
        return result




    
    
    
    
    
    