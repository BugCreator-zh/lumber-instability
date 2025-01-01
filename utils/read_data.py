from torch.utils.data import Dataset
from .yolov4.yolo import YOLO as yolo
import os
from PIL import Image
import numpy as np
import itertools
import torch

def collate_fn(data):  # 这里的data是一个list， list的元素是元组，元组构成为(self.data, self.label)
	# collate_fn的作用是把[(data, label),(data, label)...]转化成([data, data...],[label,label...])
	# 假设self.data的一个data的shape为(channels, length), 每一个channel的length相等,data[索引到数据index][索引到data或者label][索引到channel]
    
    id_list = [i[0] for i in data]
    label_list = torch.FloatTensor([i[2] for i in data])
    sag_list = [i[1][1] for i in data]
    tra_list = [i[1][0] for i in data]

    return (id_list,(tra_list,sag_list),label_list)

class SMdataset_for_pretraining(Dataset):
    def __init__(self, path,illness,config):
        self.config = config
        data0 = []
        data1 = []
        for x in os.listdir(path+'\\'+illness+'0'):
            data0.append((path+'\\'+illness+'0\\'+x,0))
        for x in os.listdir(path+'\\'+illness+'1'):
            data0.append((path+'\\'+illness+'1\\'+x,1))
        
        self.data = data0+data1
        
        plain_data = [(x,self.config.plain_transforms) for x in self.data]
        rotate_data = [(x,self.config.rotate_transforms) for x in self.data]
        
        self.total_data = plain_data + rotate_data
        
    def __getitem__(self, index):
        (img_path,img_label),transforms = self.total_data[index]
        img = Image.open(img_path)
        img = transforms(img)
        
        return (img,), img_label

    def __len__(self):
        return len(self.total_data)


class SMdataset(Dataset):
    def __init__(self, config,data,unlabled_data,sag_list,augmentation = False):
       self.config = config
       self.sag_list = sag_list
       self.labelled = data
       self.unlabelled = unlabled_data
       tmp = []
       if augmentation:
        for x in self.labelled:
            for m in range(8):
                if x[1] == 2:
                    for i in range(3):
                        tmp.append(x)
                tmp.append(x)
       self.labelled = self.labelled + tmp
       self.samples = self.labelled
        
    def __getitem__(self, index):
        img_path,img_label,transform = self.samples[index]
        if img_label == 4:
            sag_list = []
            tra_list = []
            for i in range(2):
                sag = torch.eye(256).unsqueeze(0).unsqueeze(0).repeat(len(self.sag_list),1,1,1)
                for t,x in enumerate(self.sag_list):
                    for m in os.listdir(img_path + '//' + 'SAG'):
                        if m[0:5] == x:
                            img = Image.open(img_path + '//' + 'SAG'+'//'+m)
                            sag[t] = transform(img)

                tra = transform(Image.open(img_path + '//' + 'TRA'+'//'+ os.listdir(img_path + '//' + 'TRA')[0]))

                sag_list.append(sag)
                tra_list.append(tra)

            sample_id = img_path.split('//')[-1]
            return sample_id,(tuple(tra_list),tuple(sag_list)), img_label

        sag = torch.eye(256).unsqueeze(0).unsqueeze(0).repeat(len(self.sag_list),1,1,1)
        for t,x in enumerate(self.sag_list):
            for m in os.listdir(img_path + '//' + 'SAG'):
                if m[0:5] == x:
                    img = Image.open(img_path + '//' + 'SAG'+'//'+m)
                    sag[t] = transform(img)

        tra = transform(Image.open(img_path + '//' + 'TRA'+'//'+ os.listdir(img_path + '//' + 'TRA')[0]))
        sample_id = img_path.split('//')[-1]
        return sample_id,(tra,sag), img_label

    def __len__(self):
        return len(self.samples)

    def switch_to_three_classification(self):
        f = open(self.config.Li_list)
        Li_list = ''.join(f.readlines()).split('\n')
        for i in range(len(self.samples)):
            if self.samples[i][0].split('//')[-1] in Li_list:
                self.samples[i][1] = 2


def straitified_split(samples_p,samples_n,train_split,config):
    samples_p1 = []
    samples_p2 = []
    f = open(config.Li_list)
    Li_list = ''.join(f.readlines()).split('\n')
    for i in range(len(samples_p)):
        if samples_p[i][0].split('//')[-1] in Li_list:
            samples_p2.append(samples_p[i])
        else:
            samples_p1.append(samples_p[i])

    p1_index = set(range(len(samples_p1)))
    p2_index = set(range(len(samples_p2)))
    n_index = set(range(len(samples_n)))
    train_p1_index = set(np.random.choice(list(p1_index),int(train_split*len(p1_index)),replace=False))
    train_p2_index = set(np.random.choice(list(p2_index),int(train_split*len(p2_index)),replace=False))
    train_n_index = set(np.random.choice(list(n_index),int(train_split*len(n_index)),replace=False))
    test_p1_index = p1_index - train_p1_index
    test_p2_index = p2_index - train_p2_index
    test_n_index = n_index - train_n_index

    train_p1 = [samples_p1[x] for x in list(train_p1_index)]
    train_p2 = [samples_p2[x] for x in list(train_p2_index)]
    train_n = [samples_n[x] for x in list(train_n_index)]

    test_p1 = [samples_p1[x] for x in list(test_p1_index)]
    test_p2 = [samples_p2[x] for x in list(test_p2_index)]
    test_n = [samples_n[x] for x in list(test_n_index)]

    print("Train negative samples:",len(train_n))
    print("Train positive_1 samples:",len(train_p1))
    print("Train positive_2 samples:",len(train_p2))
    print("Test negative samples:",len(test_n))
    print("Test positive_1 samples:",len(test_p1))
    print("Test positive_2 samples:",len(test_p2))    

    return train_p1+train_p2+train_n, test_p1+test_p2+test_n


def load_dataset(config,sag_list,train_split,from_trained = False):
    data_path =  config.data_path
    save_path = config.tmp_save_path
    if not from_trained:
        yolo_tra = yolo('TRA')
        yolo_sag = yolo('SAG')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for i in os.listdir(data_path):
            i_path = data_path + '\\' + i
            if not os.path.exists(save_path+'\\'+i):
                os.mkdir(save_path+'\\'+i)
            
            for t in os.listdir(i_path + '\\' +'SAG'):
                sag_save_path = save_path+'\\'+i+'\\'+'SAG'
                if not os.path.exists(sag_save_path):
                    os.mkdir(sag_save_path)
                
                img = Image.open(i_path+'\\' +'SAG'+'\\'+t)
                yolo_sag.detect_image(img,sag_save_path,crop = True)
        
            tra_save_path = save_path+'\\'+i+'\\'+'TRA'
            if not os.path.exists(tra_save_path):
                os.mkdir(tra_save_path)
            
            for t in os.listdir(i_path + '\\' +'TRA'):
                img = Image.open(i_path+'\\' +'TRA'+'\\'+t)
                yolo_tra.detect_image(img,tra_save_path,crop = True,img_type = 'TRA')
    data_path = save_path
    index = {'L5-S1.jpg','L4-L5.jpg','L3-L4.jpg','L2-L3.jpg','L1-L2.jpg'}
    incomplete = set()
    for x in os.listdir(data_path):
        if len(index -  set(os.listdir(data_path+'//'+x+'//'+'SAG'))) != 0:
               print(x,' missing SAG')
               incomplete.update({x})
        if len(os.listdir(data_path+'//'+x+'//'+'TRA'))==0:
            print(x,' missing TRA')
            incomplete.update({x})
    complete_data = list(set(os.listdir(data_path)) - incomplete)
    samples_p = []
    samples_n = []
    smaples_l = []
    samples_u = []
    for x,i in enumerate(complete_data):
        if i[0:6] == 'normal':
           samples_n.append([data_path+'//'+i,0,config.color_transforms])
        elif i[0:10] == 'unlabelled':
           samples_u.append([data_path+'//'+i,3,config.color_transforms])   
        else:
           samples_p.append([data_path+'//'+i,1,config.color_transforms])

    train_data,test_data = straitified_split(samples_p,samples_n,train_split,config)

    return SMdataset(config,train_data,samples_u,sag_list,augmentation = True),SMdataset(config,test_data,samples_u,sag_list,augmentation = True)
    
    
class SIMdataset(Dataset):
    def __init__(self, config,data,sag_list,augmentation = False):
        self.config = config
        self.data = data
        self.sag_list = sag_list
        
        data0 = []
        data1 = []
        for i in data:
            if i[1] == 0:
                data0.append(i)
            else:
                data1.append(i)

        pos_pair1 = itertools.combinations(data1, 2)
        pos_pair1 = itertools.product(pos_pair1, [1])
        pos_pair1 = list(pos_pair1)
        
        pos_pair2 = itertools.combinations(data0, 2)
        pos_pair2 = itertools.product(pos_pair2, [1])
        pos_pair2 = list(pos_pair2)
        
        pos_pair = pos_pair1 + pos_pair2
        
        neg_pair = itertools.product(data0, data1)
        neg_pair = list(neg_pair)
        neg_pair = itertools.product(neg_pair, [0])
        neg_pair = list(neg_pair)
        
        self.data = neg_pair + pos_pair
        
    def __getitem__(self, index):
         ((img_path1,img_label1,transform1),(img_path2,img_label2,transform2)),label = self.data[index]
         
         sag1 = {}
         for x in os.listdir(img_path1 + '//' + 'SAG'):
             if x[0:5] in self.sag_list:
                 img = Image.open(img_path1 + '//' + 'SAG'+'//'+x)
                 sag1[x[0:5]] = transform1(img)
         tra1 = transform1(Image.open(img_path1 + '//' + 'TRA'+'//'+ os.listdir(img_path1 + '//' + 'TRA')[0]))
         sample_id1 = img_path1.split('//')[-1]

         sag2 = {}
         for x in os.listdir(img_path2 + '//' + 'SAG'):
             if x[0:5] in self.sag_list:
                 img = Image.open(img_path2 + '//' + 'SAG'+'//'+x)
                 sag2[x[0:5]] = transform1(img)
         tra2 = transform2(Image.open(img_path2 + '//' + 'TRA'+'//'+ os.listdir(img_path2 + '//' + 'TRA')[0]))
         sample_id2 = img_path2.split('//')[-1]

         return (sample_id1,tra1,sag1),(sample_id2,tra2,sag2),label 

    def __len__(self):
        return len(self.data)





    