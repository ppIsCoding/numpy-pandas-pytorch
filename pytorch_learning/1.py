from torch.utils.data import Dataset
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获得该类别下所有图片的路径
        self.img_list = os.listdir(self.path)
    
    def __getitem__(self, index):
        # 获得该类别下第index张图片的路径
        img_name = self.img_list[index]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path)
        return img, self.label_dir
    
    def __len__(self):
        # 返回该类别下所有图片的数量
        return len(self.img_list)


# 获取蚂蚁数据集
root_dir = r"dataset\hymenoptera_data\train"
label_dir = "ants_img"
ants_dataset = MyDataset(root_dir, label_dir)

# 获取蜜蜂数据集
root_dir = r"dataset\hymenoptera_data\train"
label_dir = "bees_img"
bees_dataset = MyDataset(root_dir, label_dir)

# 合并ants_dataset和bees_dataset
train_dataset = ants_dataset + bees_dataset
