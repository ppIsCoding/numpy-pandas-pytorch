# 非线性激活
import torch
from torch import nn, tensor
from torch._dynamo import step_unsupported
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

input = torch.tensor([[1,-0.5],
[-1,3]])
input = torch.reshape(input, (-1, 1,2,2))

class Activation(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x = self.relu(x)
        x = self.sigmoid(x)
        return x

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

# print(input)
# activation = Activation()
# output = activation(input)
# print(output)
activation = Activation()
writer = SummaryWriter("p10")
step = 0
for data in dataloader:
    imgs, targets = data
    output = activation(imgs)
    writer.add_images("2026/3/8 12:06", output, step)
    step += 1

writer.close()