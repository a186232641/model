import pandas as pd
data=pd.read_excel('全国主要城市空气质量.xlsx')
data = data.fillna(-1)
# 选取性别为女性的行
df=data.loc[data['城市'] == '上海']
print(df.columns)
print(df.head())
df.to_excel('上海天气.xlsx')