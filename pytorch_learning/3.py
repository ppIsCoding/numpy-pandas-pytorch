#transform的使用  可以用于对图片进行变换
from ctypes import WinError
from tkinter import W
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter

img_path = r"dataset\hymenoptera_data\train\ants_img\0013035.jpg"
img = Image.open(img_path)
# cv_img = cv2.imread(img_path)
print(img)
# print(cv_img)

writer = SummaryWriter("logs")

# ToTensor 转换为张量
tensor_img = transforms.ToTensor()(img)
# tensor_img_cv = transforms.ToTensor()(cv_img)
print(tensor_img)
# print(tensor_img_cv)

writer.add_image("ants", tensor_img, 2)

# Normalize 归一化
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
img_norm = normalize(tensor_img)
print(img_norm)
writer.add_image("ants_norm", img_norm, 3)


# Resize 调整大小
resize = transforms.Resize((224, 224))
img_resize = resize(img)
print(img_resize)
writer.add_image("ants_resize", transforms.ToTensor()(img_resize), 4)

# Compose 组合多个变换
compose = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img_compose = compose(img)
print(img_compose)
writer.add_image("ants_compose", img_compose, 5)

# RandomCrop 随机裁剪 就是从图片中随机裁剪出一个区域
# 我们不仅要关注方法，还要关注输入以及输出的格式等
# 要多去查看官方文档  https://pytorch.org/vision/stable/transforms.html


writer.close()
