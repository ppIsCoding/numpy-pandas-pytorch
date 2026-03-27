# pandas 如何选择数据
import numpy as np
import pandas as pd

dates = pd.date_range('20260101', periods=6)
print(dates)
print("============================")
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df['A'],df.A)  # 列选择
print("============================")
print(df[0:3])  # 行选择
print(df['20260101':'20260103'])
print("============================")
# 根据标签（行和列）进行选择
print(df.loc['20260101'])
print(df.loc[:,['A','B']])
print("============================")
#根据位置
print(df.iloc[3])
print(df.iloc[[1,2,3],[1,2,3]])
print("============================")
#筛选
print(df[df.A > 0])
print("===============================")
#设置值
df.loc['20260101','C'] = 0
df.iloc[0,0] = 0
print(df)
df['D'] = np.nan
print(df)
df['E'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20260101', periods=6))
print(df)