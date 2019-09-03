# -*- coding: utf-8 -*
import numpy
import os
import torch
from flyai.model.base import Base
from path import MODEL_PATH
from transformation import src
import numpy as np
__import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
device = torch.device(device)

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

class Model(Base):
    def __init__(self, data):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)
        self.net.eval()
        x_data = self.data.predict_data(**data)
        x_data=x_data.to(device)
        # x_data = torch.from_numpy(x_data)
        outputs = self.net(x_data)
        prediction = outputs.argmax(1)
        prediction=prediction.item()
        # prediction = outputs.data.numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.net_path)
        self.net.eval()
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            # x_data = torch.from_numpy(x_data)
            x_1=src(x_data,224).to(device)
            output = self.net.forward_1(x_1)
            pre_1 = output.softmax(1).max(1)
            del x_1

            x_2=src(x_data,299).to(device)
            output = self.net.forward_2(x_2)
            pre_2 = output.softmax(1).max(1)
            del x_2

            x_3=src(x_data,224).to(device)
            output = self.net.forward_3(x_3)
            pre_3 = output.softmax(1).max(1)
            del x_3
            prediction=vote(pre_1,pre_2,pre_3).tolist()
            prediction = self.data.to_categorys(prediction)
            labels+=prediction
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))
