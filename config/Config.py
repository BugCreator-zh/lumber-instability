import torch
import torchvision.transforms as transforms 
import os
class Config():
    def __init__(self):
        root = os.getcwd()
        self.model_save_path = root+'\\logs\\'
        #self.data_path = root+'\\data\\total_data_additional_negatives'
        self.data_path = root+'\\data\\total_data_additional_negatives'
        self.Li_list = root+'\\data\\腰椎不稳.txt'
        #self.data_path = root+"\\data\\515_dataset\\LDS"
        self.label_path = ''
        self.tmp_save_path = root+'\\data\\cropped_data'
        self.embbeding_save_path = root+'\\embedding_imgs\\'
        self.Epoch = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.plain_transforms = transforms.Compose([
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.Resize([256,256]),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
        self.rotate_transforms = transforms.Compose([
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.RandomRotation(45,expand = True),
                                   transforms.Resize([256,256]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                    ])
        
        self.color_transforms = transforms.Compose([
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.ColorJitter(brightness=0.5,contrast = 0.5,hue = 0.5),
                                   transforms.Resize([256,256]),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                                    ])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        