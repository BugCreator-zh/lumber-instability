import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.evaluate import evaluation as eval_tool
from utils.evaluate import three_classification_evaluation as eval_tool1
from loss.loss_fn import deepsvdd, cr_loss
import os
import time
import copy
import matplotlib.pyplot as plt

def train(config, model,train_iter, dev_iter,optimizer = None,record = True):
    evaluation_train = eval_tool1(config,model,training = True,record = record) #三分类时使用eval_tool1
    loss_fn = F.cross_entropy
    for epoch in range(config.Epoch):
        evaluation_train.reset()
        model.train()
        for i, (sample_id,(tra,sag), y) in enumerate(train_iter):
            x1 = [m.to(config.device) for m in sag]
            x2 = [m.to(config.device) for m in tra]
            y = y.to(config.device).long()
            optimizer.zero_grad()
            outputs = model(x1,x2)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            evaluation_train.update(sample_id,loss,model.hidden_state,outputs,y)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(y),
                                                                           len(train_iter.dataset),
                                                                           100. * i / len(train_iter),
                                                                           loss.item()))
        
        val_loss,evaluation = evaluate(config,model,dev_iter,evaluation_train.save_path,epoch,loss_fn,record)
        evaluation_train.conclude_epoch(epoch,model,val_loss,evaluation,record)
    if record:
        evaluation_train.conclude()
    
    return evaluation_train.best_model,evaluation_train.dev_best_loss

def evaluate(config,model,dev_iter,save_path = None,epoch = -1,loss_fn = None,record = True):
    model.eval()
    evaluation_test = eval_tool1(config,model,save_path,training = False,record = record) ##三分类时使用eval_tool1
    with torch.no_grad():
        for sample_id,(tra,sag),y in dev_iter:
            x1 = [i.to(config.device) for i in sag]
            x2 = [i.to(config.device) for i in tra]
            y = y.to(config.device).long()
            outputs = model(x1,x2)
            loss = loss_fn(outputs, y)
            evaluation_test.update(sample_id,loss,model.hidden_state,outputs,y)
    info = evaluation_test.conclude_epoch(epoch = epoch,record = record)
    return evaluation_test.epoch_loss,info

def pl_method(config, model,train_iter, dev_iter,optimizer = None, epoches = 1,thresh = 0.8):
    loss_history = []
    time_id = '_'.join('_'.join(str(time.ctime()).split()).split(':'))
    save_path = config.model_save_path + "PL_train_" + time_id
    os.mkdir(save_path)
    tmp_train = copy.deepcopy(train_iter)
    tmp_test = copy.deepcopy(dev_iter)
    model,loss = train(config, model, tmp_train, tmp_test,optimizer, record = False)
    for t in range(epoches):
        with torch.no_grad():
            tmp_iter = copy.deepcopy(train_iter)
            tmp_iter.dataset.samples = tmp_iter.dataset.unlabelled
            new_data = []
            for i in range(tmp_iter.dataset.__len__()):
                sample_id,(tra,sag),y = tmp_iter.dataset.__getitem__(i)
                x1 = [sag.to(config.device)]
                x2 = [tra.to(config.device)]
                outputs = model(x1,x2)
                for x in enumerate(outputs[0]):
                    if x[1] > thresh:
                        single_data = tmp_iter.dataset.samples[i]
                        single_data[1] = x[0]
                        new_data.append(single_data)
        print(f'adding new samples : {str(len(new_data))}')
        tmp_train = copy.deepcopy(train_iter)
        tmp_test = copy.deepcopy(dev_iter)
        tmp_train.dataset.samples = tmp_train.dataset.samples+ new_data
        model,loss = train(config, model, tmp_train, tmp_test,optimizer, record = False)
        loss_history.append(loss)
        torch.save(model.state_dict(), save_path + "//" + str(t)+'_'+str(loss))
        print('--------------------------epoch = ',str(t),'  model loss: ',str(loss))
    plt.plot(list(range(epoches)),loss_history)
    plt.legend(['model_loss'])
    plt.savefig(save_path+'//model_loss_history.jpg')




def deep_svdd_train(config, model,train_iter, dev_iter):
    evaluation_train = eval_tool(config,model,training = True)
    #optimizer = Adam(model.parameters(),lr=0.001, betas = (0.9,0.999),eps = 1e-08,weight_decay = 1e-6,amsgrad = False)
    optimizer = torch.optim.Adagrad(model.parameters(),lr= 0.01,lr_decay=0.01,weight_decay=1e-4)
    loss_fn = deepsvdd(config,train_iter, model, eps=0.1)
    for epoch in range(config.Epoch):
        evaluation_train.reset()
        for i, (sample_id,(tra,sag), y) in enumerate(train_iter):
            x1 = [i.to(config.device) for i in sag]
            x2 = [i.to(config.device) for i in tra]
            y = y.to(config.device)
            model.train()
            optimizer.zero_grad()
            outputs = model(x1,x2)
            loss = loss_fn.compute(outputs, y)
            loss.backward()
            optimizer.step()
            outputs = loss_fn.eval_outputs(outputs)
            evaluation_train.update(sample_id,loss,model.hidden_state,outputs,y)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(y),
                                                                           len(train_iter.dataset),
                                                                           100. * i / len(train_iter),
                                                                           loss.item()))
        val_loss,evaluation = deep_svdd_evaluate(config,model,dev_iter,evaluation_train.save_path,epoch,loss_fn)
        evaluation_train.conclude_epoch(epoch,model.state_dict(),val_loss,evaluation)
    evaluation_train.conclude()
    return None

def deep_svdd_evaluate(config,model,dev_iter,save_path = None,epoch = -1,loss_fn = None):
    model.eval()
    evaluation_test = eval_tool(config,model,save_path,training = False)
    with torch.no_grad():
        for sample_id,(tra,sag),y in dev_iter:
            x1 = [i.to(config.device) for i in sag]
            x2 = [i.to(config.device) for i in tra]
            y = y.to(config.device)
            outputs = model(x1,x2)
            loss = loss_fn.compute(outputs, y)
            outputs = loss_fn.eval_outputs(outputs)
            evaluation_test.update(sample_id,loss,model.hidden_state,outputs,y)
    info = evaluation_test.conclude_epoch(epoch = epoch)
    return evaluation_test.epoch_loss,info

def SM_train(config, model,train_iter, dev_iter):
    evaluation_train = eval_tool(config,model,training = True)
    #optimizer = Adam(model.parameters(),lr=0.001, betas = (0.9,0.999),eps = 1e-08,weight_decay = 1e-6,amsgrad = False)
    optimizer = torch.optim.Adagrad(model.parameters(),lr= 0.01,lr_decay=0.01,weight_decay=1e-4)
    for epoch in range(config.Epoch):
        evaluation_train.reset()
        for i, ((sample_id1,tra1,sag1),(sample_id2,tra2,sag2),label) in enumerate(train_iter):
            x1_1 = [i.to(config.device) for i in sag1]
            x1_2 = [i.to(config.device) for i in tra1]
            x2_1 = [i.to(config.device) for i in sag2]
            x2_2 = [i.to(config.device) for i in tra2]
            y = label.to(config.device)
            model.train()
            optimizer.zero_grad()
            outputs = model((x1_1,x1_2),(x2_1,x2_2))
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            optimizer.step()
            sample_id = [ sample_id1[i] + '-' + sample_id2[i] for i in range(len(sample_id1))]
            evaluation_train.update(sample_id,loss,model.hidden_state,outputs,y)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(y),
                                                                           len(train_iter.dataset),
                                                                           100. * i / len(train_iter),
                                                                           loss.item()))
        val_loss,evaluation = SM_evaluate(config,model,dev_iter,evaluation_train.save_path,epoch)
        evaluation_train.conclude_epoch(epoch,model.state_dict(),val_loss,evaluation)
    evaluation_train.conclude()
    return None

def SM_evaluate(config,model,dev_iter,save_path = None,epoch = -1):
    model.eval()
    evaluation_test = eval_tool(config,model,save_path,training = False)
    with torch.no_grad():
        for (sample_id1,tra1,sag1),(sample_id2,tra2,sag2),label in dev_iter:
            x1_1 = [i.to(config.device) for i in sag1]
            x1_2 = [i.to(config.device) for i in tra1]
            x2_1 = [i.to(config.device) for i in sag2]
            x2_2 = [i.to(config.device) for i in tra2]
            y = label.to(config.device)
            outputs = model((x1_1,x1_2),(x2_1,x2_2))
            loss = F.cross_entropy(outputs, y)
            sample_id = [ sample_id1[i] + '-' + sample_id2[i] for i in range(len(sample_id1))]
            evaluation_test.update(sample_id,loss,model.hidden_state,outputs,y)
    info = evaluation_test.conclude_epoch(epoch = epoch)
    return evaluation_test.epoch_loss,info

def cr_train(config, model,train_iter, dev_iter):
    evaluation_train = eval_tool(config,model,training = True)
    optimizer = torch.optim.Adagrad(model.parameters(),lr= 0.01,lr_decay=0.01,weight_decay=1e-4)
    
    for epoch in range(config.Epoch):
        evaluation_train.reset()
        for i, (sample_id,(tra,sag), y) in enumerate(train_iter):
            y = y.to(config.device).long()
            model.train()
            optimizer.zero_grad()
            outputs = torch.FloatTensor([]).to(config.device)
            for z in range(len(y)):
                if y[z] == 2:
                    out1 = model(sag[z][0].to(config.device).unsqueeze(0),tra[z][0].to(config.device).unsqueeze(0))[0][0]
                    out2 = model(sag[z][1].to(config.device).unsqueeze(0),tra[z][1].to(config.device).unsqueeze(0))[0][0]
                    outputs = torch.cat([outputs,torch.FloatTensor([out1,out2]).to(config.device).unsqueeze(0)])
                    
                else:
                    outputs = torch.cat([outputs,model(sag[z].to(config.device).unsqueeze(0),tra[z].to(config.device).unsqueeze(0))])
                
            loss_fn = cr_loss
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            if len(y[torch.where(y!=2)]) != 0:
                evaluation_train.update([sample_id[x] for x in torch.where(y!=2)[0]],loss,model.hidden_state,outputs[torch.where(y!=2)],y[torch.where(y!=2)])
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(y),
                                                                           len(train_iter.dataset),
                                                                           100. * i / len(train_iter),
                                                                           loss.item()))
        val_loss,evaluation = evaluate(config,model,dev_iter,evaluation_train.save_path,epoch,loss_fn)
        evaluation_train.conclude_epoch(epoch,model.state_dict(),val_loss,evaluation)
    evaluation_train.conclude()
    return None

def cr_evaluate(config,model,dev_iter,save_path = None,epoch = -1,loss_fn = None):
    model.eval()
    evaluation_test = eval_tool(config,model,save_path,training = False)
    with torch.no_grad():
        for sample_id,(tra,sag),y in dev_iter:
            x1 = [i.to(config.device) for i in sag]
            x2 = [i.to(config.device) for i in tra]
            y = y.to(config.device)
            outputs = model(x1,x2)
            loss = loss_fn(outputs, y)
            evaluation_test.update(sample_id,loss,model.hidden_state,outputs,y)
    info = evaluation_test.conclude_epoch(epoch = epoch)
    return evaluation_test.epoch_loss,info

    