# pandas 如何处理缺失数据
import numpy as np
import pandas as pd

dates = pd.date_range('20260101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=list('ABCD'), dtype=np.int64)
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)
# 删除缺失数据 会删除行/列
print(df.dropna(axis=0, how='any')) # how = any / all  删除策略，any 表示只要有缺失数据就删除，all 表示所有数据都缺失才删除
# 填充缺失数据
print(df.fillna(value=0))
# 判断是否缺失数据
print(pd.isnull(df))
print(np.any(df.isnull()) ==  True)
