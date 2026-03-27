# numpy数据计算——2
import numpy as np

a = np.arange(2,14).reshape(3,4)

print(a)
print(np.argmin(a))  #输出数组的最小值索引
print(np.argmax(a)) #输出数组的最大值索引

# mean() median() 函数 可以计算数组的平均值和中位数
# average() 函数 可以计算数组的平均值
# cumsum() 函数 可以计算数组的累计和
# diff() 函数 可以计算数组的差分
# nonzero() 函数 输出数组的非零元素索引
# sort() 函数 可以对数组进行排序
# transpose() 函数 可以对数组进行转置  或者可以用 a.T
# clip() 函数 可以对数组进行截断 可以设置上下限，小于或大于限制的元素会被限制在限制之内