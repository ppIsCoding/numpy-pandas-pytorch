import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    # 创建模型实例
    model = CIFAR10Classifier()
    
    input_tensor = torch.ones((64, 3, 32, 32))
    
    # 将输入传入模型进行前向传播
    output = model(input_tensor)
    
    # 打印输出形状：应该是 [64, 10]，表示64个样本，每个样本10个类别的得分
    print(output.shape)
