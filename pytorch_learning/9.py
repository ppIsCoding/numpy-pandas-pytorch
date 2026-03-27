# 池化层  
# 池化操作就是对输入的特征图进行下采样，减少特征图的大小，同时保留重要的特征信息。
# 找到特征图中每个区域的最大值或平均值，作为该区域的输出。
import weakref
import torch
from torch import nn, tensor
from torch._dynamo import step_unsupported
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=64,
    )

# input = torch.tensor([[1,2,0,3,1],
#                      [0,1,2,3,1],
#                      [1,2,1,0,0],
#                      [5,2,3,1,1],
#                      [2,1,0,1,1],
# ])

# input = torch.reshape(input, (-1, 1, 5, 5))

class Pooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
    
    def forward(self, x):
        x = self.pool(x)
        return x

# pooling = Pooling()
# output = pooling(input)
# print(output)

writer = SummaryWriter("p10")

step = 0
for data in dataloader:
    imgs, targets = data
    # 写入tensorboard
    writer.add_images("2026/3/7 17:39", imgs, step)
    step += 1

writer.close()  

