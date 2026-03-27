# 关于Tensorboard的使用 图形变换 以及 可视化模型 tansform的使用
from torch.utils.tensorboard import SummaryWriter
# 还可以用OpenCV去读取图片
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

img_path = r"dataset\hymenoptera_data\train\ants_img\0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(img_array.shape)

# 参数：
# 第一个参数：标题 第二个参数：图片数组 第三个参数：全局步长 第四个参数：数据格式
writer.add_image("ants", img_array, 1,dataformats="HWC")
# y = x
# for i in range(100):
#     # 第一个参数：标题 第二个参数：y轴 第三个参数：x轴
#     writer.add_scalar("y=x", i, i)
# writer.add_scalar()

writer.close()