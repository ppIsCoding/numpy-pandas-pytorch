# 介绍numpy的计算
import numpy as np

a = np.array([10, 20, 30])
b = np.arange(3)

c = a - b # 数减法
print(c)
d = a + b # 数加法
print(d)
e = a * b  # 数乘法
print(e)
f = 10 * np.sin(a) # 使用了特殊函数
print(f)
print(f < 3)  # 数比较
g = np.dot(a, b) # 矩阵乘法
print(g)

h = np.random.random((2,3)) # 创建一个随机数组
print(h)

# np.sum() .min() .max() 函数 可以对数组进行求和、求最小值、求最大值
# 可以添加一下参数 axis=0 表示对横轴进行操作 axis=1 表示对纵轴进行操作