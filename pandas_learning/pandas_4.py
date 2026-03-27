# 导入 导出数据
import pandas as pd

data = pd.read_csv('student.csv')
print(data)

data.to_csv('student_new.csv')