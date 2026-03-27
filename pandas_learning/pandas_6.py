#  merge
import pandas as pd
import numpy as np

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                       'A': ['A0', 'A1', 'A2', 'A3'],
                       'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})

res = pd.merge(left, right, on='key')  # 基于 key 进行合并
# how = {‘left', ‘right’, 'outer', 'inner'} 四种选择。默认为 inner
# indicator = True 添加合并的标志列
# left_index 和 right_index 基于索引进行合并
# suffixes = ('_x', '_y') 添加列的标识符

print("合并结果:")
print(res)
 