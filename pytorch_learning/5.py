#  DataLoader的使用
# 一些常见参数：
# dataset：数据集对象
# batch_size：每个批次的样本数量
# shuffle：是否在每个 epoch 开始时打乱数据
# num_workers：使用多少个子进程加载数据 如果有问题 
# drop_last：是否在最后一个批次中丢弃不足 batch_size 的样本
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=data_transforms, download=True)

test_loader = DataLoader(
    dataset=test_set,
     batch_size=4, 
     shuffle=True, 
     num_workers=0,
     drop_last=False)

# 测试集中第一张图片以及对应的target
img, target = test_set[0]  # 调用__getitem__方法
print(img.shape)
print(target)

writer = SummaryWriter("p10")

# 测试数据加载器
step = 0
for data in test_loader:
    # 每个批次的图片和标签
    imgs, targets = data
    # 打印每个批次的图片和标签
    print(imgs.shape)
    # 打印每个批次的标签
    print(targets)
    # 写入tensorboard
    writer.add_images("test_set", imgs, step)
    step += 1

writer.close()
