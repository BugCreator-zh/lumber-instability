import time
import os
import torch
import pandas as pd
import numpy as np
from .visualize import plot_tsne3, show_conclusion
from ignite.metrics import Accuracy
from ignite.metrics import Precision
from ignite.metrics import Recall
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

class evaluation():
    def __init__(self,config,model,save_path = None,training = False,record = True):
        self.tag = 'train' if training else 'test'
        self.config = config
        self.training = training
        if training and record:
            time_id = '_'.join('_'.join(str(time.ctime()).split()).split(':'))
            self.save_path = config.model_save_path + "train_" + time_id
            os.mkdir(self.save_path)
        else:
            self.save_path = save_path
            self.inputs = torch.FloatTensor([]).to(config.device)
            
        self.train_loss_history = []
        self.train_score_history = []
        self.epoch_loss = 0
        self.test_score = []
        self.batch_num = 0
        self.dev_best_loss = float('inf')
        self.best_model = None

        self.embeddings = torch.FloatTensor([]).to(config.device)
        self.targets = torch.IntTensor([]).to(config.device)

        self.test_acc = Accuracy()
        self.test_recall = Recall()
        self.test_precision = Precision()

        self.result = pd.DataFrame(columns = ['sample_id','ground_truth','prediction'])

    def reset(self):
        self.epoch_loss = 0
        self.batch_num = 0
        self.embeddings = torch.FloatTensor([]).to(self.config.device)
        self.targets = torch.IntTensor([]).to(self.config.device)
        self.test_precision.reset()
        self.test_acc.reset()
        self.test_recall.reset()
        self.result = pd.DataFrame(columns = ['sample_id','ground_truth','prediction'])
        

    def update(self,sample_id,loss,hidden_state,outputs,y):
        if not self.training:
            self.inputs = torch.cat([self.embeddings,hidden_state])
        self.epoch_loss = self.epoch_loss + loss.item()
        self.batch_num = self.batch_num + 1
        
        self.embeddings = torch.cat([self.embeddings,hidden_state])
        self.targets = torch.cat([self.targets,torch.unsqueeze(y,dim = 1)])

        self.test_acc.update((outputs, y))
        self.test_recall.update((outputs, y))
        self.test_precision.update((outputs, y))

        #记录结果
        tmp_1 = outputs.cpu().detach().numpy()
        tmp_2 = y.cpu().detach().numpy()
        tmp_3 = np.array([ x[0] for x in tmp_1])
        sample_id = np.array(sample_id)
        tmp = np.array([sample_id,tmp_2,tmp_3])
        self.result = pd.concat([self.result,pd.DataFrame(tmp.T,columns = ['sample_id','ground_truth','prediction'])])

    def conclude_epoch(self,epoch = -1,model=None,val_loss=None,evaluation = None,record = True):
        total_acc = self.test_acc.compute()
        total_recall = self.test_recall.compute()
        total_precision = self.test_precision.compute()
        info = 'acc_'+str(int(total_acc*1000)/1000)+'_recall_'+str(int(total_recall[0].item()*1000)/1000)+','+str(int(total_recall[1].item()*1000)/1000)+'_precision_'+str(int(total_precision[0].item()*1000)/1000)+','+str(int(total_precision[1].item()*1000)/1000)
        print(f'---------------------------------------------- {self.tag}  avg_loss = {self.epoch_loss/self.batch_num}    info: {info}')

        if record == False:
            if self.tag == 'train':
                if val_loss < self.dev_best_loss:
                    self.dev_best_loss = val_loss
                    self.best_model = model
            return None

        plot_tsne3(self.embeddings.cpu().detach().numpy(),self.targets.cpu().detach().numpy(),self.config.embbeding_save_path,name1=self.tag)

        if epoch == -1:
            self.result.to_csv(self.save_path + '//' + self.tag+'_'+info + '.csv')
        else:
            self.result.to_csv(self.save_path + '//' + self.tag+'_'+str(epoch)+'_'+info + '.csv')
        
        if self.tag == 'train':
            self.train_loss_history.append(self.epoch_loss/self.batch_num)
            self.train_score_history.append(float(info.split('_')[1]))
            self.test_score.append(float(evaluation.split('_')[1]))

            torch.save(model.state_dict(), self.save_path + "//" + str(epoch)+'_'+evaluation)
            if val_loss < self.dev_best_loss:
                self.dev_best_loss = val_loss
                self.best_model = model
                torch.save(model.state_dict(), self.save_path + '//best_trained_model')

        return info   
                    
    def conclude(self):
        show_conclusion(self.train_loss_history,self.train_score_history,self.test_score,self.save_path+'//history.jpg')
        
class three_classification_evaluation():
    def __init__(self,config,model,save_path = None,training = False,record = True):
        self.tag = 'train' if training else 'test'
        self.config = config
        self.training = training
        if training:
            time_id = '_'.join('_'.join(str(time.ctime()).split()).split(':'))
            self.save_path = config.model_save_path + "train_" + time_id
            os.mkdir(self.save_path)
        else:
            self.save_path = save_path
            self.inputs = torch.FloatTensor([]).to(config.device)
            
        #self.model = SmoothGradCAMpp(model)
        self.train_loss_history = []
        self.train_score_history = []
        self.epoch_loss = 0
        self.test_score = []
        self.batch_num = 0
        self.dev_best_loss = float('inf')
        self.best_model = None

        self.embeddings = torch.FloatTensor([]).to(config.device)
        self.targets = torch.IntTensor([]).to(config.device)

        self.test_acc = Acc(config)

        self.result = pd.DataFrame(columns = ['sample_id','ground_truth','prediction'])

    def reset(self):
        self.epoch_loss = 0
        self.batch_num = 0
        self.embeddings = torch.FloatTensor([]).to(self.config.device)
        self.targets = torch.IntTensor([]).to(self.config.device)
        self.test_acc.reset()
        self.result = pd.DataFrame(columns = ['sample_id','ground_truth','prediction'])
        

    def update(self,sample_id,loss,hidden_state,outputs,y):
        if not self.training:
            self.inputs = torch.cat([self.embeddings,hidden_state])
        self.epoch_loss = self.epoch_loss + loss.item()
        self.batch_num = self.batch_num + 1
        
        self.embeddings = torch.cat([self.embeddings,hidden_state])
        self.targets = torch.cat([self.targets,torch.unsqueeze(y,dim = 1)])

        self.test_acc.update(outputs, y)

        #记录结果
        tmp_1 = outputs.cpu().detach().numpy()
        tmp_2 = y.cpu().detach().numpy()
        tmp_3 = np.array([ str(x[0]) + '_' + str(x[1]) + '_' + str(x[2]) for x in tmp_1])
        sample_id = np.array(sample_id)
        tmp = np.array([sample_id,tmp_2,tmp_3])
        self.result = pd.concat([self.result,pd.DataFrame(tmp.T,columns = ['sample_id','ground_truth','prediction'])])

    def conclude_epoch(self,epoch = -1,model=None,val_loss=None,evaluation = None,record = True):
        total_acc = self.test_acc.compute()  
        info = 'TotalAcc_'+str(int(total_acc[0]*1000)/1000)+'_acc1_'+str(int(total_acc[1]*1000)/1000)+'_acc2_'+str(int(total_acc[2]*1000)/1000)+'_acc3_'+str(int(total_acc[3]*1000)/1000)
        print(f'---------------------------------------------- {self.tag}  avg_loss = {self.epoch_loss/self.batch_num}    info: {info}')
        if record == False:
            if self.tag == 'train':
                if val_loss < self.dev_best_loss:
                    self.dev_best_loss = val_loss
                    self.best_model = model
            return None

        plot_tsne3(self.embeddings.cpu().detach().numpy(),self.targets.cpu().detach().numpy(),self.config.embbeding_save_path,name1=self.tag)

        if epoch == -1:
            self.result.to_csv(self.save_path + '//' + self.tag+'_'+info + '.csv')
        else:
            self.result.to_csv(self.save_path + '//' + self.tag+'_'+str(epoch)+'_'+info + '.csv')
        
        if self.tag == 'train':
            self.train_loss_history.append(self.epoch_loss/self.batch_num)
            self.train_score_history.append(float(info.split('_')[1]))
            self.test_score.append(float(evaluation.split('_')[1]))

            torch.save(model.state_dict(), self.save_path + "//" + str(epoch)+'_'+evaluation)
            if val_loss < self.dev_best_loss:
                self.dev_best_loss = val_loss
                torch.save(model.state_dict(), self.save_path + '//best_trained_model')

        return info   
                    
    def conclude(self):
        show_conclusion(self.train_loss_history,self.train_score_history,self.test_score,self.save_path+'//history.jpg')
    
class Acc():
    def __init__(self,config):
        self.config = config
        self.outputs = torch.FloatTensor([]).to(config.device)
        self.y = torch.IntTensor([]).to(config.device)

    def reset(self):
        self.outputs = torch.FloatTensor([]).to(self.config.device)
        self.y = torch.IntTensor([]).to(self.config.device)

    def update(self,outputs,y):
        self.outputs = torch.cat([self.outputs,outputs])
        self.y = torch.cat([self.y,y])
    def compute(self):
        num = [0,0,0]
        acc = [0,0,0]
        for i in range(len(self.y)):
            label = int(torch.where(self.outputs[i] == max(self.outputs[i]))[0])
            num[self.y[i]] = num[self.y[i]] + 1
            if label == int(self.y[i]):
                acc[self.y[i]] = acc[self.y[i]] + 1
            
        result = [0,0,0,0]
        result[0] = (acc[0]+acc[1]+acc[2])/(num[0]+num[1]+num[2])
        result[1] = acc[0]/num[0]
        result[2] = acc[1]/num[1]
        result[3] = acc[2]/num[2]
        return result



        

