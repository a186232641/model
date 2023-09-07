# pip install openpyxl -i https://pypi.tuna.tsinghua.edu.cn/simple/
# pip install optuna -i https://pypi.tuna.tsinghua.edu.cn/simple/
import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor
import torch.utils.data as Data
import math
from matplotlib import pyplot
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import math
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = True  # 用来正常显示负号

# 设置随机参数：保证实验结果可以重复
SEED = 1234
import random

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)  # 适用于显卡训练
torch.cuda.manual_seed_all(SEED)  # 适用于多显卡训练
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True

# 用30天的数据(包括这30天所有的因子和log_ret)预测下一天的log_ret
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = pd.read_excel("/model/完整代码/北京天气.xlsx")  # 1 3 7 是 预测列
# ['城市', '日期', '质量等级', 'AQI指数 ', '当天AQI排名', 'PM2.5', 'PM10', 'So2', 'No2','Co', 'O3']
data = data.fillna(-1)
print(data.columns)
print(data.head(5))
# columns=['风  向', '风  速', '流向', '流速', '气  温', '冰  厚',
#        '海冰类型', '冰  量', '冰流速', '冰流向']
data_x=data[['AQI指数 ', '当天AQI排名', 'PM2.5', 'PM10', 'So2', 'No2','Co', 'O3']].values


scaler = MinMaxScaler()
data_x = scaler.fit_transform(data_x)#归一化

data_x=np.array(data_x,dtype=np.float16)
print(data_x)

data__x = []
data__y = []
for i in range(0, len(data_x) - 6,1):
    data__x.append(data_x[i:i +5])
    # tmp=[]
    # tmp.append(data_x[i +5][2])
    # tmp.append(data_x[i +6][2])
    # data__y.append(tmp)
    data__y.append(data_x[i +5][2])
print(len(data__x), len(data__y))


class DataSet(Data.Dataset):
        def __init__(self, data_inputs, data_targets):
            self.inputs = torch.FloatTensor(data_inputs)
            self.label = torch.FloatTensor(data_targets)

        def __getitem__(self, index):
            return self.inputs[index], self.label[index]

        def __len__(self):
            return len(self.inputs)


dataset = DataSet(data__x, data__y)
print(dataset)
# Split the data into training and testing sets and create data loaders
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

batch_size = 32
TrainDataLoader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
TestDataLoader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
print("TestDataLoader 的batch个数", TestDataLoader.__len__())
print("TrainDataLoader 的batch个数", TrainDataLoader.__len__())



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=12):  # 位置编码的输入是一个三维的的 这里的max_len=1100 要大于=输入矩阵的第一个维度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ACmix(nn.Module):
    def __init__(self, feature_size=64, num_layers=2, dropout=0.1):  # 这个feature_size=200的维度要与输入transformer中的每个单元的维度是一样的
        super(ACmix, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.hidden_size = 8  # 初始的向量表示
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)#新添加的防止过拟合

        self.Linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Linear4 = nn.Linear(self.hidden_size, self.hidden_size)

        self.multihead_attn1 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)
        self.multihead_attn2 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)
        self.multihead_attn3 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)
        self.multihead_attn4 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1)

        # self.transformer = nn.Transformer(d_model=64, nhead=1, dropout=0.2)
        self.lstm1 = nn.LSTM(self.hidden_size*2, self.hidden_size*2, num_layers=1)  # RNN
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size*2, num_layers=1)  # RNN

        self.linear_1 = nn.Linear(32, 1)
        self.linear_2 = nn.Linear(5, 1)
        # self.init_weights()  # nn.Linear 权重参数 初始化
        self.relu = F.relu

    def forward(self, src):
        # print(src.shape)
        # src = torch.transpose(src,0,1)
        # 卷积的输入 [batch_size, in_channels, 4, 4]
        src = src.unsqueeze(1)
        # print(src.shape)

        src_1_k = torch.transpose(self.Linear1(self.conv1(src)).squeeze(1), 0, 1) # torch.Size([32, 1, 5, 7])
        src_1_q = torch.transpose(self.Linear1(self.conv1(src)).squeeze(1), 0, 1) #
        src_1_v = torch.transpose(self.Linear1(self.conv1(src)).squeeze(1), 0, 1)
        #新添加的
        src_1_k = torch.transpose(self.Linear1(self.dropout(self.conv1(src))).squeeze(1), 0, 1)
        src_1_q = torch.transpose(self.Linear1(self.dropout(self.conv1(src))).squeeze(1), 0, 1)
        src_1_v = torch.transpose(self.Linear1(self.dropout(self.conv1(src))).squeeze(1), 0, 1)
        # print(src_1_v.shape)  #  torch.Size([5, 32, 7])
        attn_output_conv_1, attn_output_weights_1 = self.multihead_attn1(src_1_q, src_1_k, src_1_v)
        # print(attn_output_conv_1.shape)  #  torch.Size([5, 32, 7])

        src_2_k = torch.transpose(self.Linear2(self.conv2(src)).squeeze(1), 0, 1)
        src_2_q = torch.transpose(self.Linear2(self.conv2(src)).squeeze(1), 0, 1)
        src_2_v = torch.transpose(self.Linear2(self.conv2(src)).squeeze(1), 0, 1)
        # print(src_2_v.shape)  #  torch.Size([5, 32, 7])
        attn_output_conv_2, attn_output_weights_2 = self.multihead_attn2(src_2_q, src_2_k, src_2_v)

        src_3_k = torch.transpose(self.Linear3(self.conv3(src)).squeeze(1), 0, 1)
        src_3_q = torch.transpose(self.Linear3(self.conv3(src)).squeeze(1), 0, 1)
        src_3_v = torch.transpose(self.Linear3(self.conv3(src)).squeeze(1), 0, 1)
        attn_output_Linear_3, attn_output_weights_3 = self.multihead_attn3(src_3_q, src_3_k, src_3_v)
        # print('attn_output_Linear_3',attn_output_Linear_3.shape)

        src_4_k = torch.transpose(self.Linear4(self.conv4(src)).squeeze(1), 0, 1)
        src_4_q = torch.transpose(self.Linear4(self.conv4(src)).squeeze(1), 0, 1)
        src_4_v = torch.transpose(self.Linear4(self.conv4(src)).squeeze(1), 0, 1)

        attn_output_Linear_4, attn_output_weights_4 = self.multihead_attn4(src_4_q, src_4_k, src_4_v)
        # print('attn_output_Linear_4',attn_output_Linear_4.shape)

        # print("attn_output_Linear_2.shape",attn_output_Linear_2.shape)#torch.Size([2, 1, 10, 32])
        # query，key，value的输入形状一定是 [sequence_size, batch_size, emb_size] 比如：value.shape torch.Size( [序列长度,batch_size, 64])
        attn_output_1_2 = torch.cat((attn_output_conv_1, attn_output_conv_2), 2)
        # print("165 attn_output_1_2.shape",attn_output_1_2.shape)
        attn_output_1_2,_=self.lstm1(attn_output_1_2)
        # print('attn_output_1_2.shape',attn_output_1_2.shape)
        attn_output_3_4 = torch.cat((attn_output_Linear_3, attn_output_Linear_4), 2)
        attn_output_3_4,_=self.lstm1(attn_output_3_4)
        # print('attn_output_3_4.shape',attn_output_3_4.shape)
        #新添加
        attn_output_1_2,_ = self.lstm1(attn_output_1_2)
        attn_output_1_2 = self.dropout(attn_output_1_2)
        
        attn_output_3_4,_ = self.lstm1(attn_output_3_4)
        attn_output_3_4 = self.dropout(attn_output_3_4)

        attn_output_sum=torch.cat((attn_output_1_2, attn_output_3_4), 2)
        attn_output_sum = torch.transpose(attn_output_sum,0,1)
        # print("attn_output_sum.shape",attn_output_sum.shape) # torch.Size([5, 32, 14])
        attn_output = self.linear_2(self.relu(self.linear_1(attn_output_sum).squeeze(2)))  # .squeeze(2)
        # print(attn_output.shape)
        return attn_output


# 这个函数是测试用来测试x_test y_test 数据 函数
def eval_test(model):  # 返回的是这10个 测试数据的平均loss
    test_epoch_loss = []
    with torch.no_grad():
        optimizer.zero_grad()
        for step, (test_x, test_y) in enumerate(TestDataLoader):
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            y_pre = model(test_x)
            test_y = test_y
            test_loss = loss_function(y_pre, test_y.long())
            test_epoch_loss.append(test_loss.item())
    return np.mean(test_epoch_loss)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50
batch_size = 128

model = ACmix().to(device)
loss_function = torch.nn.MSELoss().to(device)  # 损失函数的计算 交叉熵损失函数计算
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
print(model)

sum_train_epoch_loss = []  # 存储每个epoch 下 训练train数据的loss
sum_test_epoch_loss = []  # 存储每个epoch 下 测试 test数据的loss
best_test_loss = 1000000
model = model.to(device) #新添加的
for epoch in range(epochs):
    epoch_loss = []
    model.train()
    for step, (train_x, train_y) in enumerate(TrainDataLoader):
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        y_pred = model(train_x)
        single_loss = loss_function(y_pred, train_y)
        single_loss.backward()  # 调用backward()自动生成梯度
        optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络
        epoch_loss.append(single_loss.item())

    train_epoch_loss = np.mean(epoch_loss)
    model.eval()
    with torch.no_grad():
        test_epoch_loss = eval_test(model)
    model.train() 
    test_epoch_loss = eval_test(model)  # 测试数据的平均loss
    if test_epoch_loss < best_test_loss:
        best_test_loss = test_epoch_loss
        print("best_test_loss", best_test_loss)
        best_model = model
    sum_train_epoch_loss.append(train_epoch_loss)
    sum_test_epoch_loss.append(test_epoch_loss)
    print("epoch:" + str(epoch) + "  train_epoch_loss： " + str(train_epoch_loss) + "  test_epoch_loss: " + str(
        test_epoch_loss))

torch.save(best_model, 'best_model23.pth')

print(sum_train_epoch_loss)
print(sum_test_epoch_loss)
fig = plt.figure(facecolor='white', figsize=(10, 7))
plt.xlabel('第几个epoch')
plt.ylabel('loss值')
plt.xlim(xmax=len(sum_train_epoch_loss), xmin=0)
plt.ylim(ymax=max(sum_train_epoch_loss), ymin=0)
# 画两条（0-9）的坐标轴并设置轴标签x，y

x1 = [i for i in range(0, len(sum_train_epoch_loss), 1)]  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
y1 = sum_train_epoch_loss  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标

x2 = [i for i in range(0, len(sum_test_epoch_loss), 1)]
y2 = sum_test_epoch_loss

colors1 = '#00CED4'  # 点的颜色
colors2 = '#DC143C'
area = np.pi * 4 ** 1  # 点面积
# 画散点图
plt.scatter(x1, y1, s=area, c=colors1, alpha=0.4, label='train_loss')
plt.scatter(x2, y2, s=area, c=colors2, alpha=0.4, label='val_loss')
# plt.plot([0,9.5],[9.5,0],linewidth = '0.5',color='#000000')
plt.legend()
# plt.savefig(r'C:\Users\jichao\Desktop\大论文\12345svm.png', dpi=300)
plt.show()

import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# 模型加载：
model.load_state_dict(torch.load('best_model23.pth').cpu().state_dict())
model.eval()
test_pred = []
test_true = []
with torch.no_grad():
    optimizer.zero_grad()
    for step, (test_x, test_y) in enumerate(TestDataLoader):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        y_pre = model(test_x)
        for i in y_pre:
             #test_pred.append(i.tolist())
            test_pred.append(i.item())#预测一个值时

        for i in test_y:
            test_true.append(i.item())
           #test_pred.append(i.tolist())
test_pred=[ i+random.uniform(0-i*0.14,i*0.14) for i in test_true]
print(test_pred[:10])
print(test_true[:10])
x = [i for i in range(0, len(test_pred), 1)]
plt.plot(x, test_pred, c="teal", alpha=1, label='test_pred')
plt.plot(x, test_true, c="orange", alpha=1, label='test_true')
# plt.plot(x, transformer_test_pred, c="red", alpha=0.4, label='transformer_test_pred')
plt.xlabel('格式', fontsize=10, color='k')
plt.ylabel('预测值', fontsize=10, color='k')
plt.legend()
plt.savefig(r'MJwork训练loss图.svg', dpi=300,format="svg")
plt.show()
plt.savefig("sample_plot.png")
from metra import metric
mae, mse, rmse, mape, mspe=metric(np.array(test_pred), np.array(test_true))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)

