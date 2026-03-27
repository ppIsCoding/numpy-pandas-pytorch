# 完整的训练过程（基于CIFAR10数据集）

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from model import *

# 添加 tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./logs")

# 设置使用第一个 GPU (NVIDIA RTX)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
if torch.cuda.is_available():
    print(f"GPU 名称：{torch.cuda.get_device_name(0)}")

# 准备数据集
train_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', 
        train=True, 
        transform=torchvision.transforms.ToTensor(), 
        download=True)

test_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', 
        train=False, 
        transform=torchvision.transforms.ToTensor(), 
        download=True)

# 输出一下数据集的长度
print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 加载数据集
train_loader = DataLoader(
    dataset=train_dataset,
     batch_size=64,)

test_loader = DataLoader(
    dataset=test_dataset,
     batch_size=64,)


# 搭建神经网络 10 分类的模型 并且在 GPU 上运行
classifier = CIFAR10Classifier()
classifier.to(device)


# 损失函数 并且在 GPU 上计算
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)


# 优化器
learning_rate = 1e-2
optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)

# 训练参数 训练网络参数
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮数
epoch = 10

for i in range(epoch):
    print(f"第 {i + 1} 轮训练开始...")

    classifier.train()
    # 训练步骤开始
    for data in train_loader:
        imgs, targets = data
        # 将数据放入 GPU
        imgs = imgs.to(device)
        targets = targets.to(device)
        # 训练
        outputs = classifier(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"第 {total_train_step} 步：{loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)


    total_test_loss = 0
    total_accuracy = 0
    classifier.eval()
    with torch.no_grad():
        # 测试步骤开始
        for data in test_loader:
            imgs, targets = data
            # 将数据放入 GPU
            imgs = imgs.to(device)
            targets = targets.to(device)
            # 前向传播
            outputs = classifier(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print(f"第 {i + 1} 轮结束，测试集整体损失值为：{total_test_loss}")
    print(f"第 {i + 1} 轮结束，测试集整体准确率值为：{total_accuracy / len(test_dataset):.2f}")
    writer.add_scalar("test_accuracy", total_accuracy / len(test_dataset), total_test_step + 1)
    writer.add_scalar("test_loss", total_test_loss, total_test_step + 1)
    total_test_step += 1

    # 保存模型
    torch.save(classifier.state_dict(), f"cifar10_model_{i + 1}.pth")
    print(f"第 {i + 1} 轮模型保存完成")

writer.close()