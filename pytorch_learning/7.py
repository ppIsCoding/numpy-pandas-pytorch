# conv2d
import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                        [0,1,0],
                        [2,1,0]])

# 对于我们的输入和卷积核，我们需要将它们转换为4维张量
# 第一个维度是批量大小，这里我们只有一个样本，所以是1
# 第二个维度是通道数，这里我们只有一个通道，所以是1
# 第三个维度是高度，这里是5
# 第四个维度是宽度，这里是5
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input, kernel, stride=1)
print(output)

