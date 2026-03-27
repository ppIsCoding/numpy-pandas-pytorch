# numpy 数组复制
import numpy as np

a = np.arange(4)
b = a
print(a)
print(b)

a[0] = 5
print(a)
print(b)
print(b is a)
print("====================================")
c = a.copy()  # deep  copy
print(c is a)