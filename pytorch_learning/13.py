from torchvision import datasets, transforms
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
# 网络模型的使用
# 1. 加载预训练模型
# 2. 修改模型的最后一层

# train_data = torchvision.datasets.ImageNet("./dataset", 
#                     split="train",
#                     transform=transforms.ToTensor(), 
#                     download=True)


# 二者的区别：
# 1. 是否加载预训练模型
# 2. 预训练模型的下载地址
# 3. 预训练模型的参数是否冻结
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./dataset", 
                    train=True,
                    transform=transforms.ToTensor(), 
                    download=True)

vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)
