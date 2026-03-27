# numpy 索引
import numpy as np

a = np.arange(3,15)
print(a[0]) #一维

a = a.reshape(3,4)
print(a[0]) #二维
print(a[0][0])
print(a[0,0])

print(a[:,0])
print("===============================================")
for row in a: # 迭代行
    print(row)
print("===============================================")
for column in a.T: # 迭代列
    print(column)
print("===============================================")
print(a.flatten())
for item in a.flat: # 迭代所有元素
    print(item)