import argparse
import os
import numpy as np
import math
import time
import taichi
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
taichi.init()
# 创建一个名为images的目录，并使用argparse模块创建一个命令行解释器
os.makedirs("images", exist_ok=True)
parser = argparse.ArgumentParser()
# 使用argparse模块为一个图像生成器模型的训练过程定义了多个命令行参数
# opt.n_epochs表示训练的epoch数量
# batch_size表示每个批次的大小
# opt.1r表示Adam优化器的学习率
# opt.b1和opt.b2分别表示Adam优化器的两个衰减因子
# opt.n_cpu表示用于批次生成的CPU线程数量
# opt.latent_dim表示生成器模型中的潜在空间的维度
# opt.n_classes表示数据集中的类别数量
# opt.img_size表示生成的图像的大小
# opt.channels表示生成的图像的通道数
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)
# 定义一个元组
img_shape = (opt.channels, opt.img_size, opt.img_size)
# 检查当前计算机是否支持CUDA
cuda = True if torch.cuda.is_available() else False

# 生成器模块
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 创建了一个标签嵌入层
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        # in_feat:输入特征的数量 out_feat:输出特征的数量 normalize:表示是否应该使用批归一化
        def block(in_feat, out_feat, normalize=True):
            # 创建一个nn.Linear层
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                # 添加nn.BatchNorm1d层，将输出特征进行批归一化
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
# 定义了一个由多个层组成的序列模型。包括标签嵌入层、全连接层、批归一化层和激活函数
        self.model = nn.Sequential(
            # 接受随机噪声向量和标签嵌入向量的连接作为输入，并将其映射到一个128维的向量
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            # 将128维的向量映射到一个256维的向量
            *block(128, 256),
            # 将256维的向量映射到一个512维的向量
            *block(256, 512),
            *block(512, 1024),
            # 使用nn.Linear层将1024维的向量映射到一个大小等于图像形状维度乘积的向量
            # 再使用双曲正切激活函数进行转化以生成最终的生成图像
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    # 将随机噪声和标签向量转换为一个生成的图像
    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        # 将向量通过生成器的序列模型进行处理
        img = self.model(gen_input)
        # 将生成的向量重新变形为符合图像形状的张量
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 将标签向量编码成密集的向量表示
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        # 用于训练判别器，以识别生成器生成的假图像和真实图像的区别
        self.model = nn.Sequential(
            # 将标签向量和图像展平后的向量连接起来，并将其映射到一个512维的向量
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            # 对线性层的输出进行非线性变换
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            # 随机丢弃一些神经元，以减少过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        # 将图像张量展平为一个向量，并将标签向量通过标签嵌入层进行编码
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        # 将输入向量映射到一个标量输出，表明输入图像是否真实
        validity = self.model(d_in)
        return validity

# 创建一个MSEloss对象，用于计算生成器和判别器的对抗损失函数
# Loss functions
adversarial_loss = torch.nn.MSELoss()
# 创建一个Generator对象和一个Discriminator对象
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
# 如果cuda为True 就启用GPU加速
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        # 应用一个图像变换pipeline
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
# 定义两个优化器使用Adam优化算法来更新生成器和判别器的参数
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# 创建两个张量类型，FLoatTensor和LongTensor 用于数据和模型参数的移动
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# 生成和保存生成的图像的函数
def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        # 从输入图像的形状中获取当前批次的大小
        batch_size = imgs.shape[0]

        # 创建一个大小与输入图像相同的全为1的张量。此张量代表真实图像的真实标签
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # 将输入的真实图像转换为FloatTensor，并包装为PyTorch变量
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------
        # 初始化生成器网络的梯度为0
        optimizer_G.zero_grad()

        # 从均值为0、标准差为1的正态分布中随机采样噪声
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # 用生成器生成一批图像
        gen_imgs = generator(z, gen_labels)

        # 用鉴别器判别生成器生成的图像的真实性，并计算生成器的损失
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        # 反向传播生成器的损失
        g_loss.backward()
        # 更新生成器的权重
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # 用鉴别器判别真实图像的真实性
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # 用鉴别器判别生成的图像的真实性并计算生成的图像的损失
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # 计算总的鉴别器损失
        d_loss = (d_real_loss + d_fake_loss) / 2

        # 反向传播鉴别器的损失
        d_loss.backward()
        # 更新鉴别器的权重
        optimizer_D.step()
        # 打印每个批次的鉴别器和生成器损失，并显示当前训练的周期数和批次数
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
digit = input("Please input the required number:")
maxnumber = epoch * 468
img = Image.open('./images/' + str(maxnumber) + '.png')
# 定义裁剪区域
left = int(34.2 * (int(digit) ))
right = int(left + 34.2)
crop_area = (left, 0, right, 34)
cropped_img = img.crop(crop_area)
timestamp = int(time.time())
timestamp_str = str(timestamp)
filename = './obj_images/cropped_image_{}.png'.format(timestamp_str)
cropped_img.save(filename)
cropped_img.show()
