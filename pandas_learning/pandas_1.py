# pandas介绍
import numpy as np
import pandas as pd

s = pd.Series([1, 3, 5, np.nan, 6, 8]) # 创建一个Series
print(s)
print("============================")
dates = pd.date_range('20260101', periods=6)
print(dates)
print("============================")
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
# index 和 columns 是DataFrame的索引和列标签 可以不设定  数据也可以使用一个字典来创建
df2 = pd.DataFrame({'A': 1.,
                     'B': pd.Timestamp('20130102'),
                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                     'D': np.array([3] * 4, dtype='int32'),
                     'E': pd.Categorical(["test", "train", "test", "train"]),
                     'F': 'foo'})
print(df)
print(df2)
# dtypes 显示数据类型 index 索引 columns 列标签 values 显示数据 T 列转行
# describe 描述性统计(只能对数型数据) sort_index sort_values 分别是对索引和列进行排序