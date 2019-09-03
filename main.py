# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: user
"""
from sklearn.metrics import f1_score
import argparse
import torch
from flyai.dataset import Dataset
from model import Model
from net import Net
from path import MODEL_PATH
import math
from torch.optim import Adadelta,Adam,SGD
import torch.nn as nn
from transformation import src,change_1,change_2,batch_Y


'''
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

'''
项目的超参
'''
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=6, type=int, help="batch size")
args = parser.parse_args()

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def vote(pre_1,pre_2,pre_3):
    pre_1_conf, pre_1_cls = pre_1
    pre_2_conf, pre_2_cls = pre_2
    pre_3_conf, pre_3_cls = pre_3

    result=[]
    for i in range(pre_1_conf.size(0)):
        a=pre_1_conf[i].item()
        b=pre_2_conf[i].item()
        c=pre_3_conf[i].item()
        if a>=b and a>=c:
            result.append(pre_1_cls[i])
        elif b>=a and b>=c:
            result.append(pre_2_cls[i])
        elif c>=a and c>=b:
            result.append(pre_3_cls[i])
        else:
            result.append(pre_3_cls[i])
    return np.array(result)

'''
flyai库中的提供的数据处理方法
传入整个数据训练多少轮，每批次批大小
'''
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
dataset.test_trans=True
model = Model(dataset)
'''
实现自己的网络机构
'''
label_list = ['apron', 'bare-land', 'baseball-field', 'basketball-court', 'beach', 'bridge', 'cemetery', 'church', 'commercial-area', 'desert', 'dry-field', 'forest', 'golf-course', 'greenhouse', 'helipad', 'ice-land', 'island', 'lake', 'meadow', 'mine', 'mountain', 'oil-field', 'paddy-field', 'park', 'parking-lot', 'port', 'railway', 'residential-area', 'river', 'road', 'roadside-parking-lot', 'rock-land', 'roundabout', 'runway', 'soccer-field', 'solar-power-plant', 'sparse-shrub-land', 'storage-tank', 'swimming-pool', 'tennis-court', 'terraced-field', 'train-station', 'viaduct', 'wind-turbine', 'works']
num_classes=len(label_list)
# 判断gpu是否可用
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

#squeezenet1_1,squeezenet1_0,shufflenetv2_x1.0,shufflenetv2_x0.5,inception_v3_google,
# resnet18,densenet121,densenet161,densenet169,densenet201

model_list=['resnet18','inception_v3_google','densenet121']
# model_list=['squeezenet1_0','inception_v3_google','resnet18','densenet121']
# model_list=['densenet121']
net = Net(model_list,num_classes=num_classes).to(device)
criterions=[nn.CrossEntropyLoss().to(device)]*len(model_list)
lrs=[1e-4,1e-4,1e-4]
optimizers=[Adam(params=net.model_list[ii].parameters(), lr=lrs[ii], weight_decay=1e-5) for ii in range(len(model_list))]
'''
dataset.get_step() 获取数据的总迭代次数

'''

chang_lr=[14,18,19]

val_iter=50
show_iter=50
best_score = 0
steps=dataset.get_step()
chang_lr_=[math.ceil(float(s)/args.EPOCHS*steps) for s in chang_lr]
chang_lr=chang_lr_
print(steps)
print(chang_lr)

test_dui=0
test_num=0
test_acc=0.0
test_f1=0.0
train_f1=0.0
train_i=0
test_i=0
train_num=0
models_log={'model_1':{'train':{'loss':0.0,'acc':0.0,'f1':0.0},'test':{'loss':0.0,'acc':0.0,'f1':0.0}},
            'model_2': {'train':{'loss':0.0,'acc':0.0,'f1':0.0},'test':{'loss':0.0,'acc':0.0,'f1':0.0}},
            'model_3': {'train':{'loss':0.0,'acc':0.0,'f1':0.0},'test':{'loss':0.0,'acc':0.0,'f1':0.0}},}

all_acc={'train':0.0,'test':0.0}

for step in range(steps):
    if step in chang_lr:
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr']*0.1
    net.train()
    x_train, y_train,x_val,y_val = dataset.next_batch()
    x_train_1=change_1(x_train,224).to(device)
    y_train=batch_Y(y_train).to(device)

    train_num+=x_train_1.size(0)

    ###模型一的情况
    output=net.forward_1(x_train_1)
    loss=criterions[0](output,y_train)
    optimizers[0].zero_grad()
    loss.backward()
    optimizers[0].step()
    pre_1 = output.softmax(1).max(1)
    models_log['model_1']['train']['loss']+=float(loss.item())
    models_log['model_1']['train']['acc']+=float((pre_1[1]==y_train).sum().item())
    models_log['model_1']['train']['f1'] += f1_score(y_train.cpu().tolist(),pre_1[1].cpu().tolist(),average='macro')
    del x_train_1

    #模型二的情况
    x_train_2=change_2(x_train,299).to(device)
    output=net.forward_2(x_train_2)
    loss = criterions[1](output, y_train)
    optimizers[1].zero_grad()
    loss.backward()
    optimizers[1].step()
    pre_2 = output.softmax(1).max(1)
    models_log['model_2']['train']['loss']+=float(loss.item())
    models_log['model_2']['train']['acc']+=float((pre_2[1]==y_train).sum().item())
    models_log['model_2']['train']['f1'] += f1_score(y_train.cpu().tolist(),pre_2[1].cpu().tolist(),average='macro')
    del x_train_2


    #模型三的情况
    x_train_3=src(x_train,224).to(device)
    output=net.forward_3(x_train_3)
    loss = criterions[2](output, y_train)
    optimizers[2].zero_grad()
    loss.backward()
    optimizers[2].step()
    pre_3 = output.softmax(1).max(1)
    models_log['model_3']['train']['loss']+=float(loss.item())
    models_log['model_3']['train']['acc']+=float((pre_3[1]==y_train).sum().item())
    models_log['model_3']['train']['f1'] += f1_score(y_train.cpu().tolist(),pre_3[1].cpu().tolist(),average='macro')
    train_i+=1
    del x_train_3

    pre=vote(pre_1,pre_2,pre_3)
    gt=y_train.cpu().numpy()
    all_acc['train']+=(pre==gt).sum()

    del y_train
    del pre_1
    del pre_3
    del pre_2

    if (step+1)%val_iter==0 or step==steps-1:
        net.eval()

        ##模型一的测试情况
        x_val_1=src(x_val,224).to(device)
        y_val=batch_Y(y_val).to(device)
        output = net.forward_1(x_val_1)
        pre_1=output.softmax(1).max(1)
        loss=criterions[0](output, y_val)
        models_log['model_1']['test']['loss'] += float(loss.item())
        models_log['model_1']['test']['acc'] += float((pre_1[1] == y_val).sum().item())
        models_log['model_1']['test']['f1'] += f1_score(y_val.cpu().tolist(), pre_1[1].cpu().tolist(), average='macro')
        del x_val_1
        test_num += output.size(0)
        test_i+=1
        ##模型二的测试情况
        x_val_2 = src(x_val, 299).to(device)
        output=net.forward_2(x_val_2)
        pre_2 = output.softmax(1).max(1)
        loss = criterions[1](output, y_val)
        models_log['model_2']['test']['loss'] += float(loss.item())
        models_log['model_2']['test']['acc'] += float((pre_2[1] == y_val).sum().item())
        models_log['model_2']['test']['f1'] += f1_score(y_val.cpu().tolist(), pre_2[1].cpu().tolist(), average='macro')
        del x_val_2

        ##模型三的测试情况
        x_val_3 = src(x_val, 224).to(device)
        output=net.forward_3(x_val_3)
        pre_3 = output.softmax(1).max(1)
        loss = criterions[2](output, y_val)
        models_log['model_3']['test']['loss'] += float(loss.item())
        models_log['model_3']['test']['acc'] += float((pre_3[1] == y_val).sum().item())
        models_log['model_3']['test']['f1'] += f1_score(y_val.cpu().tolist(), pre_3[1].cpu().tolist(), average='macro')
        del x_val_3

        pre = vote(pre_1, pre_2, pre_3)
        gt=y_val.cpu().numpy()
        all_acc['test'] += (pre == gt).sum()
        test_acc=all_acc['test']/test_num
        if best_score< test_acc:
            best_score=test_acc
            model.save_model(net, MODEL_PATH, overwrite=True)

        del y_val
        del pre_1
        del pre_2
        del pre_3

    if (step + 1) % show_iter == 0 or step==steps-1:
        ##训练
        ###模型一
        print('##################################################################')
        print('----------------------    train    ------------------------------')
        loss=models_log['model_1']['train']['loss']/train_i
        acc=models_log['model_1']['train']['acc']/train_num
        f1=models_log['model_1']['train']['f1']/train_i
        log = "Train>>1 [{}/{}] Loss:{:.7f} Acc:{:.7f} F1:{:.7f}".format(step, steps, loss, acc,f1)
        print(log)

        ###模型二
        loss=models_log['model_2']['train']['loss']/train_i
        acc=models_log['model_2']['train']['acc']/train_num
        f1=models_log['model_2']['train']['f1']/train_i
        log = "Train>>2 [{}/{}] Loss:{:.7f} Acc:{:.7f} F1:{:.7f}".format(step, steps, loss, acc,f1)
        print(log)

        ###模型三
        loss=models_log['model_3']['train']['loss']/train_i
        acc=models_log['model_3']['train']['acc']/train_num
        f1=models_log['model_3']['train']['f1']/train_i
        log = "Train>>3 [{}/{}] Loss:{:.7f} Acc:{:.7f} F1:{:.7f}".format(step, steps, loss, acc,f1)
        print(log)

        ###投票精度
        print('Train>> Vote Acc:{:.7f}'.format(all_acc['train']/train_num))

        ##测试
        ###模型一
        print('----------------------    Test    ------------------------------')
        loss=models_log['model_1']['test']['loss']/test_i
        acc=models_log['model_1']['test']['acc']/test_num
        f1=models_log['model_1']['test']['f1']/test_i
        log = "Test>>1 [{}/{}] Loss:{:.7f} Acc:{:.7f} F1:{:.7f}".format(step, steps, loss, acc,f1)
        print(log)
        ###模型二
        loss=models_log['model_2']['test']['loss']/test_i
        acc=models_log['model_2']['test']['acc']/test_num
        f1=models_log['model_2']['test']['f1']/test_i
        log = "Test>>2 [{}/{}] Loss:{:.7f} Acc:{:.7f} F1:{:.7f}".format(step, steps, loss, acc,f1)
        print(log)

        ###模型三
        loss=models_log['model_3']['test']['loss']/test_i
        acc=models_log['model_3']['test']['acc']/test_num
        f1=models_log['model_3']['test']['f1']/test_i
        log = "Test>>3 [{}/{}] Loss:{:.7f} Acc:{:.7f} F1:{:.7f}".format(step, steps, loss, acc,f1)
        print(log)
        print("Test>> Vote Acc:{:.7f}".format(all_acc['test'] / test_num))
        print('##################################################################')
