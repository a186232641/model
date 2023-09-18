import pandas as pd

# 读取Excel文件
file_path = '/Users/hanfeilong/Desktop/模型/完整代码/北京天气.xlsx'  # 请将此路径替换为您的Excel文件路径
df = pd.read_excel(file_path)

# 将'日期'列转化为日期格式并排序
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values(by='日期')

# 保存排序后的数据到新的Excel文件
output_path = '北京天气.xlsx'  # 您可以更改此路径以指定保存位置
df.to_excel(output_path, index=False, engine='openpyxl')

print("排序完成并保存到新的Excel文件!")
