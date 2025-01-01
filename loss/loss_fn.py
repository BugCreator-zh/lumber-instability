import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F

class deepsvdd():
    def __init__(self,config,train_loader:dataloader, net, eps=0.1):
        n_samples = 0
        self.config = config
        self.c = torch.zeros(128).to(config.device)
        net.eval()
        with torch.no_grad():
            for sample_id,(tra,sag),y in train_loader:
                # get the inputs of the batch
                x1 = [i.to(config.device) for i in sag.values()]
                x2 = tra.to(config.device)
                outputs = net(x1, x2)
                n_samples += outputs.shape[0]
                b = torch.sum(outputs, dim=0)
                self.c += b
        self.c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
        self.c[(abs(self.c) < eps) & (self.c > 0)] = eps

        self.R = torch.tensor(0).to(config.device)
        self.eps = 1e-6
        self.eta = 1

    def compute(self,outputs,y):
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        losses = torch.where(y==1,dist,self.eta * ((dist + self.eps) ** y.float()))
        loss = torch.mean(losses)
        return loss

    def eval_outputs(self,outputs):
        result = torch.FloatTensor([]).to(self.config.device)
        t1 = torch.FloatTensor([[0.,1.]]).to(self.config.device)
        t2 = torch.FloatTensor([[1.,0.]]).to(self.config.device)
        for i,v in enumerate(outputs):
            r = torch.cdist(v.unsqueeze(0),self.c.unsqueeze(0),p=2)
            if r < self.R:
                result = torch.cat([result,t1],dim = 0)
            else:
                result = torch.cat([result,t2],dim = 0)
        return result
"""
class cr_loss(nn.Module):
    def __init__(self):
        
    def forward(self, outputs, y):
        u_target = outputs[torch.where(y==2)]
        l_target0 = outputs[torch.where(y==0)]
        l_target1 = outputs[torch.where(y==1)]
        
        loss_1 = torch.mean(torch.pow((u_target[:,0] - u_target[:,1]), 2))
        loss_2 = 0  
        if len(l_target0) != 0:
            loss_2 -= torch.sum(torch.log2(l_target0[:,0]))
        if len(l_target1) != 0:
            loss_2 -= torch.sum(torch.log2(l_target1[:,0]))
        return loss_1+loss_2
"""
def cr_loss(outputs, y):
        u_target = outputs[torch.where(y==2)]
        l_target0 = outputs[torch.where(y==0)]
        l_target1 = outputs[torch.where(y==1)]
        
        loss_1 = torch.mean(torch.pow((u_target[:,0] - u_target[:,1]), 2))
        loss_2 = 0  
        if len(l_target0) != 0:
            loss_2 -= torch.sum(torch.log2(l_target0[:,0]))
        if len(l_target1) != 0:
            loss_2 -= torch.sum(torch.log2(l_target1[:,0]))
        loss = loss_1+loss_2
        loss.requires_grad_(True)
        return loss 

