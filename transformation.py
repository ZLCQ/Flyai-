# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch
class Transformation:
    '''
    处理训练数据的类，某些情况下需要对训练的数据再一次的处理。
    如无需处理的话，不用实现该方法。
    '''

    def transformation_data(self, x_train=None, y_train=None, x_test=None, y_test=None):

        return x_train, y_train, x_test, y_test



def batch_X_transform(Xs,transforms):
    imgs=[]
    for x in Xs:
        img=Image.fromarray(x)
        img=transforms(img)
        imgs.append(img)
    imgs=torch.stack(imgs,dim=0)
    return imgs

def batch_Y(Ys):
    return torch.from_numpy(Ys).long()

def src(Xs,size):
    tran= transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        )])
    return batch_X_transform(Xs,tran)

def change_1(Xs,size):
    tran= transforms.Compose([
                              transforms.RandomCrop(size),
                              transforms.RandomRotation(20),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        )])
    return batch_X_transform(Xs,tran)

def change_2(Xs,size):
    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    # intensities of enhancers
    factors = {
        0: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
        1: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        2: lambda: np.clip(np.random.normal(1.0, 0.15), 0.7, 1.3),
        3: lambda: np.clip(np.random.normal(1.0, 0.3), 0.4, 1.6),
    }

    # randomly change color of an image
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        # random enhancers in random order
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image
    tran= transforms.Compose([
                              transforms.Resize(size),
                              transforms.Lambda(enhance),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]
                                        )])
    return batch_X_transform(Xs,tran)