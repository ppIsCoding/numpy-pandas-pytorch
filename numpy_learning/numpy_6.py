# numpy array的合并
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.vstack((a,b))) # 垂直合并
print("================================")
print(np.hstack((a,b))) # 水平合并
print("================================")
print(a)
print(a[np.newaxis,:])
print(a[:,np.newaxis]) # 增加一个维度
print("================================")
c = np.concatenate((a,b),axis=0) #进行多个矩阵合并，axis=0表示垂直合并，axis=1表示水平合并
print(c)