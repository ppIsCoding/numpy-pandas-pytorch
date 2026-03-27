# numpy array的分割
import numpy as np

a = np.arange(12).reshape(3,4)

print(a)
print(np.split(a,2,axis=1)) # 对数组进行分割，axis=1表示纵向分割
# 但是这个方法是只能进行等分的分割，如果不能进行等分的分割，那么就只能使用array_split方法
print(np.array_split(a,3,axis=1))

# vsplit 垂直分割 hsplit 水平分割