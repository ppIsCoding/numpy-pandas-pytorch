# 对CIFAR-10数据集进行分类
from torchvision import datasets, transforms
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5,padding=2,stride=1),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

cifar10_classifier = CIFAR10Classifier()

# 验证
# input = torch.ones((64, 3, 32, 32))
# output = cifar10_classifier(input)
# print(output.shape)

# writer = SummaryWriter("logs_seq")
# writer.add_graph(cifar10_classifier, input)
# writer.close()

dataset = datasets.CIFAR10("./dataset", train=False, transform=transforms.ToTensor(), download=True)
daraloader = torch.utils.data.DataLoader(dataset, batch_size=64)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cifar10_classifier.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in daraloader:
        inputs, targets = data
        output = cifar10_classifier(inputs)
        result_loss = loss(output, targets)
        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()
        running_loss += result_loss.item()
    print(epoch, running_loss)
    
