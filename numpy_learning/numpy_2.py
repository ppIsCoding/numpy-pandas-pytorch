# 本代码是在介绍numpy数组时使用的代码

import numpy as np

a = np.array([2,2,3],dtype=np.float64) #可以在创建的数组中指定数据类型
print(a)

b = np.zeros((3,4),dtype=np.int64)  #创建一个全零数组
print(b)

c = np.ones((3,4),dtype=np.int64)  #创建一个全1数组
print(c)

d = np.arange(0,10) #创建一个等差数列
print(d)

e = np.arange(0,10).reshape(2,5) #创建一个等差数列并改变形状
print(e)

f = np.linspace(0,10,5) #创建分段数列
print(f)