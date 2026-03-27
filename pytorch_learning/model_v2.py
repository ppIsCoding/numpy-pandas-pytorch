import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class CIFAR10ClassifierV2(nn.Module):
    def __init__(self):
        super(CIFAR10ClassifierV2, self).__init__()
        self.model = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            
            # 第二层卷积
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            
            # 第三层卷积
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            
            # 全连接层
            nn.Flatten(),
            nn.Linear(in_features=128 * 4 * 4, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=10)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    model = CIFAR10ClassifierV2()
    input_tensor = torch.ones((64, 3, 32, 32))
    output = model(input_tensor)
    print(f"输出形状：{output.shape}")
