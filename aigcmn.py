import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from infogan import Generator,Discriminator,\
    FloatTensor,opt,to_categorical


class AiGcMn:
    def __init__(self, generator_params_pth,discriminator_params_pth):
        # 加载模型、设置参数等
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_params_pth=generator_params_pth
        self.discriminator_params_pth=discriminator_params_pth

    def misc(self):
        # 其他处理函数
        self.generator.load_state_dict(torch.load(self.generator_params_pth, map_location=torch.device('cpu')))
        self.discriminator.load_state_dict(torch.load(self.discriminator_params_pth, map_location=torch.device('cpu')))

        # Static generator inputs for sampling
        static_z = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.latent_dim))))
        static_label = to_categorical(
            (np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)])), num_columns=opt.n_classes
        )
        static_code = Variable(FloatTensor(np.zeros((opt.n_classes ** 2, opt.code_dim))))

        z = Variable(FloatTensor(np.random.normal(0, 1, (opt.n_classes ** 2, opt.latent_dim))))
        self.static_sample = self.generator(z, static_label, static_code)

        zeros = np.zeros((opt.n_classes ** 2, 1))
        c_varied = np.repeat(np.linspace(-1, 1, opt.n_classes)[:, np.newaxis], opt.n_classes, 0)
        c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))
        c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))
        self.sample1 = self.generator(static_z, static_label, c1)
        self.sample2 = self.generator(static_z, static_label, c2)

    def generate(self, target):

        if isinstance(target,torch.Tensor):
            target_tensor=target
        else:
            target_tensor=torch.tensor(target)

        # 定义映射关系字典
        mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 7,
            4: 8,
            5: 4,
            6: 5,
            7: 6,
            8: 3,
            9: 9
        }

        # 创建索引张量
        input_tensor= torch.tensor([mapping[i.item()] for i in target_tensor])
        #input_tensor=target_tensor

        result = torch.empty(0)
        for i in input_tensor:
            result = torch.cat([result, self.sample2[i+30].unsqueeze(0)], dim=0)

        #print("输出tensor: ",result.size())

        # 假设生成的图像为 gen_img，类型为 torch.Tensor
        resized_result = torch.nn.functional.interpolate(result, size=(28, 28), mode='bilinear', align_corners=True)
        #print("输出修改后的tensor: ", resized_result.size())
        save_image(resized_result, "AiGcMn_test_0.png", nrow=resized_result.size(0), normalize=True)
        return resized_result





if __name__=='__main__':

    data = [0,1,2,3,4,5,6,7,8,9]

    # input为输入的整数型n维tensor
    input=torch.tensor(data)

    # 创建AiGcMn接口类的实例，需要传入2个参数(是预先训练好的模型参数)
    test=AiGcMn("generator_params.pth","discriminator_params.pth")
    # misc()为其他处理函数，用于加载和中间过程的处理
    test.misc()
    """
    接口函数generate:target参数为一个整型的n维torch.Tensor，
    （n是batch的大小，每个整数在0~9范围内，代表需要生成的数字）
    该函数返回n*1*28*28的tensor
    （n是batch的大小，每个1*28*28的tensor表示随机生成的数字图像）
    并且会在当前目录生成一行对应数字的网格图像
    """
    output_tensor=test.generate(target=input)
    print("输出图像tensor.size: ",output_tensor.size())

