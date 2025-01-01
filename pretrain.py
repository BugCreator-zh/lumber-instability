from collections import Counter
import torch
import os
import shutil
from copy import deepcopy
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
from utils.read_data import SMdataset,SMdataset_for_pretraining
from torch.utils.data import random_split
from config import Config
from ignite.metrics import Accuracy
from ignite.metrics import Precision
from ignite.metrics import Recall
from models.vggnet import VGG
from models.Siamese import Siamese_net
from models.multiple_input_net import MIN
from utils.visualize import plot_tsne3
from torchvision.models import mobilenet,resnet18
import torch.nn as nn

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

def train(config, train_iter, dev_iter):
    #model = Siamese_net(config)
    #model = MIN(config,1).to(config.device)
    #model = VGG(config,2).to(config.device)
    #model = mobilenet.MobileNetV3(2)
    model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(config.device)
    
    optimizer = SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=1e-6, nesterov=True)
    train_loss_history = []
    val_loss_history = []
    counter = []
    iteration_number = 0
    total_batch = 0  #
    last_improve = 0  #
    flag = False  #
    dev_best_loss = float('inf')
    best_model = None
    record = []
    
    test_acc = Accuracy()
    test_recall = Recall()
    test_precision = Precision()
    
    for epoch in range(config.Epoch):
        
        #embeddings = torch.FloatTensor([]).to(model.config.device)
        #targets = torch.IntTensor([]).to(model.config.device)
        
        print("---------epoch:",epoch,"----------------------")
        for i, (x, y) in enumerate(train_iter):
            print("------",i,"-------")
            x = [i.to(config.device) for i in x]
            y = y.to(config.device)
            model.train()
            optimizer.zero_grad()
            outputs = model(x[0])
            
            #embeddings = torch.cat([embeddings,x[0]])
           # targets = torch.cat([targets,y])
            
            test_acc.update((outputs, y))
            test_recall.update((outputs, y))
            test_precision.update((outputs, y))
            
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            record.extend(y.cpu().numpy().tolist())
            # 在验证集测试用于保存最佳模型
            if i % config.num_check == 0:

                #训练集loss记录
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                iteration_number += 10
                counter.append(iteration_number)
                train_loss_history.append(loss.item())

                #验证集loss记录
                val_loss = evaluate(dev_iter, model)
                val_loss_history.append(val_loss.item())

                if val_loss < dev_best_loss:
                    dev_best_loss = val_loss
                    best_model = deepcopy(model.state_dict())
                    torch.save(model.state_dict(), config.model_save_path+'best_pretrained_model')
                    last_improve = total_batch
                model.train()
            total_batch += 1
            # early stopping
            if total_batch - last_improve > config.not_improvement:
                print("No improvement for a long time, early-stopping...")
                flag = True
                break
            
        total_acc = test_acc.compute()
        total_recall = test_recall.compute()
        total_precision = test_precision.compute()
        
        print("ACC:",total_acc)
        print("recall of every test dataset class: ", total_recall)
        print("precision of every test dataset class: ", total_precision)

        test_precision.reset()
        test_acc.reset()
        test_recall.reset()
        
        #plot_tsne3(embeddings.cpu().detach().numpy(),targets.cpu().detach().numpy(),'C:\\JLUstudy\\code\\python\\project',name1="0",model_name="0")
        
        if flag:
            break
    show_plot(counter, val_loss_history)
    print(Counter(record))
    return best_model


def evaluate(dev_iter1, model):
    model.eval()
    loss_total = 0
    test_acc = Accuracy()
    test_recall = Recall()
    test_precision = Precision()
    #embeddings = torch.FloatTensor([]).to(model.config.device)
   # targets = torch.IntTensor([]).to(model.config.device)
    num=0
    with torch.no_grad():
        for x,y in dev_iter1:
            x = [i.to(config.device) for i in x]
            y = y.to(config.device)
            outputs = model(x[0])
           # embeddings = torch.cat([embeddings,model.embedding])
            #targets = torch.cat([targets,y])
            test_acc.update((outputs, y))
            test_recall.update((outputs, y))
            test_precision.update((outputs, y))
            loss = F.cross_entropy(outputs, y)
            loss_total += loss
            num=num+1
            print('----------eval ',num,' -----------')
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    
    print("ACC:",total_acc)
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)

    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    
    #plot_tsne3(embeddings.cpu().detach().numpy(),targets.cpu().detach().numpy(),'C:\\JLUstudy\\code\\python\\project',name1="0",model_name="0")
    
    return loss_total / len(dev_iter1)

if __name__ == '__main__':
    
    config = Config.Config()
    
    data = SMdataset_for_pretraining(config.data_path,'lds',config)
    
    train_num = int(4/5*data.__len__())
    test_num = data.__len__() - train_num
    train_data,test_data = random_split(data,[train_num,test_num])

    
    train_data = DataLoader(train_data, batch_size=5, shuffle=True)
    test_data = DataLoader(test_data, batch_size=5, shuffle=True)
    
    train(config, train_data, test_data)
    
    