import pydicom
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import numpy as np 
import os
import re

#图像转换
def convert(data):
    high = np.max(data)
    low = np.min(data)
    img = (data-low)/(high - low)
    img = (img*255).astype('uint8')
    img = Image.fromarray(img)
    return img
#转换并保存
def convert_save(path,img_type,img_id,save_path,crop = None):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path+'\\'+img_type
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if crop == None:
        crop = {'TRA':[30,120,310,280],'SAG':[100,50,250,380]}   #参考裁剪值 只要满足TRA:280*160
                                                                 #                  SAG:120*330
        
    for i,x in enumerate(os.listdir(path)):
        data = pydicom.read_file(path+'\\'+x).pixel_array
        img = convert(data)
        #img = img.crop(crop[img_type[-3:]])
        img.save(save_path+'\\'+str(i)+'.jpg')
        print(x+'  completed')
        
#调整某样本的某类图像的裁剪,crop值需要满足固定分辨率，输入crop为字典
#例：
"""
crop = {'TRA':[130,50,250,380]}
adjust_single('0002','T1_TSE_TRA',crop,data_path,save_path)
"""
def adjust_single(sample_id,img_type,crop,data_path,save_path): 
    path = data_path + '\\' + sample_id
    path = path + '\\'+os.listdir(path)[0]
    content = os.listdir(path)
    for i in content:
        if bool(re.match(img_type+'.*',i)):
            convert_save(path+'\\'+i,img_type,sample_id,save_path+ '\\' + sample_id)
        

data_path = 'data\\515_dataset\\MRI_Data'  #mypart改为自己的文件夹即可，该文件夹下存在的就是0001等样本文件
save_path = 'data\\515_dataset\\processed_data'

#批量预处理
for x in os.listdir(data_path):
    path = data_path+'\\'+x
    for m in os.listdir(path):
        if m[:7] == 'L-SPINE':
            path = path + '\\' +m
    content = os.listdir(path)
    for i in content:
        if bool(re.match('T1_TSE_SAG.*',i)):
            convert_save(path+'\\'+i,'T1_TSE_SAG',x,save_path+ '\\' + x)
        elif bool(re.match('T1_TSE_TRA.*',i)):
            convert_save(path+'\\'+i,'T1_TSE_TRA',x,save_path+ '\\' + x)
        elif bool(re.match('T2_TSE_SAG.*',i)):
            convert_save(path+'\\'+i,'T2_TSE_SAG',x,save_path+ '\\' + x)
        elif bool(re.match('T2_TSE_TRA.*',i)):
            convert_save(path+'\\'+i,'T2_TSE_TRA',x,save_path+ '\\' + x)

    print('---------',x,'--------')
    


    
    
    
    
    
    



