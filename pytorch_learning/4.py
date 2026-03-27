import torchvision
from torchvision.models.maxvit import F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=data_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=data_transforms, download=True)

# print(f"训练集大小: {len(train_set)}")
# print(f"测试集大小: {len(test_set)}")
# print(f"第一个样本: {train_set[0]}")
# print(f"第一个测试样本: {test_set[0]}")
# print(f"数据集类别: {train_set.classes}")

writer = SummaryWriter("p10")
for i in range(10):
    img, label = train_set[i]
    # 第一个参数：标题 第二个参数：图片数组 第三个参数：全局步长 
    writer.add_image(f"train_set", img, i)

writer.close()