# 合并
import pandas as pd
import numpy as np

# concatenating
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])

res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
print(res)
print("=================================")
# join
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
res = pd.concat([df1,df2],join='inner',ignore_index=True)
res = pd.concat([df1,df2.reindex(df1.index)],axis=1)
print( res)
print("=================================")