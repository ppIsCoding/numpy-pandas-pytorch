# 加入数据集 进行 conv2d
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=data_transforms, download=True)

dataLoader = DataLoader(
    dataset=dataset,
    batch_size=64,
    )

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        return x

model = model()

writer = SummaryWriter("p10")

step = 0
for data in dataLoader:
    imgs, targets = data
    output = model(imgs)
    print(output.shape)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output_1", output, step)
    step += 1

writer.close()

