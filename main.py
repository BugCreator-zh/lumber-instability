from torch.utils.data import DataLoader
from utils.read_data import load_dataset,collate_fn
from models.multiple_input_net import MIN
from models.Relationalnet import SM
from config import Config
from method import train,evaluate,deep_svdd_train,SM_train,cr_train,pl_method
import torch

if __name__ == '__main__':
    #导入配置文件
    config = Config.Config()
    #随机数种子设置
    #torch.manual_seed(10)
    #导入数据
    #在这里确认使用的sag样本
    sag_list = ['L3-L4','L4-L5','L5-S1']
    train_data,test_data = load_dataset(config,sag_list,train_split = 2/3,from_trained = True)
    #三分类
    train_data.switch_to_three_classification()
    test_data.switch_to_three_classification()
    #加入无标签数据
    #train_data.samples = train_data.samples + train_data.unlabelled 
    #转换
    train_data = DataLoader(train_data, batch_size=15, shuffle=True,collate_fn=collate_fn,drop_last=True)
    test_data = DataLoader(test_data, batch_size=15, shuffle=True,drop_last=True)
    #-------模型导入
    # resnet18 googlenet alexnet vggnet
    model = MIN(config,['googlenet','googlenet']).to(config.device)
    #model = SM(config,['googlenet','googlenet']).to(config.device)
    #--------------
    #导入上次训练节点的模型
    #model.load_state_dict(torch.load(config.model_save_path + 'acc_0.65',map_location=config.device))
    #优化器
    #optimizer = Adam(model.parameters(),lr=0.001, betas = (0.9,0.999),eps = 1e-08,weight_decay = 1e-6,amsgrad = False)
    optimizer = torch.optim.Adagrad(model.parameters(),lr= 0.01,lr_decay=0.01,weight_decay=1e-4)
    #训练
    train(config,model, train_data, test_data,optimizer)
    #伪标签训练，倒数第一个值为挑选无标签样本的置信度，倒数第二个为模型迭代次数（和训练轮数不同，训练轮数在config里调）
    #pl_method(config,model, train_data, test_data, optimizer, 10,0.9) 