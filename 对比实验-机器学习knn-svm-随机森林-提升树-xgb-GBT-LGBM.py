# 基于机器学习XGB svm LGBM knn的
from sklearn import preprocessing
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from datetime import datetime
import time
import math
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from scipy import stats, integrate
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 评价指标
from sklearn.linear_model import LogisticRegression
from metra import metric
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data = pd.read_excel("北京天气.xlsx")  # 1 3 7 是 预测列
# ['城市', '日期', '质量等级', 'AQI指数 ', '当天AQI排名', 'PM2.5', 'PM10', 'So2', 'No2','Co', 'O3']
data = data.fillna(-1)
print(data.columns)
print(data.head(5))
# columns=['风  向', '风  速', '流向', '流速', '气  温', '冰  厚',
#        '海冰类型', '冰  量', '冰流速', '冰流向']
data_x=data[['AQI指数 ', '当天AQI排名', 'PM2.5', 'PM10', 'So2', 'No2','Co', 'O3']].values
data_x=np.array(data_x,dtype=np.float16)
print(data_x)

data__x = []
data__y = []
for i in range(0, len(data_x) - 5,1):
    data__x.append(data_x[i:i +5])
    data__y.append(data_x[i +5][2])
print(len(data__x), len(data__y))
data__x=np.array(data__x)
data__y=np.array(data__y)
data__x=data__x.reshape(data__x.shape[0],-1)
x_train, x_test, y_train, y_test = train_test_split(np.array(data__x), np.array(data__y), test_size=0.2)

data_mse = []
data_mae = []
# knn算法
knn = KNeighborsRegressor()
knn.fit(x_train, y_train)
knn_test_pred = knn.predict(x_test)  # 进行预测
print("knn算法----------------------------------------- ")
print(knn_test_pred[:10])
print(y_test[:10])

mae, mse, rmse, mape, mspe=metric(np.array(knn_test_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)

print('mean_squared_error:', mean_squared_error(y_test, knn_test_pred))  # mse
data_mse.append(mean_squared_error(y_test, knn_test_pred))
print("mean_absolute_error:", mean_absolute_error(y_test, knn_test_pred))  # mae
data_mae.append(mean_absolute_error(y_test, knn_test_pred))

# svm算法
svm = SVR()
svm.fit(x_train, y_train)
svm_test_pred = svm.predict(x_test)
print("svm算法 ")
print(svm_test_pred[:10])
print(y_test[:10])
mae, mse, rmse, mape, mspe=metric(np.array(svm_test_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)
print('mean_squared_error:', mean_squared_error(y_test, svm_test_pred))  # mse
data_mse.append(mean_squared_error(y_test, svm_test_pred))
print("mean_absolute_error:", mean_absolute_error(y_test, svm_test_pred))  # mae
data_mae.append(mean_absolute_error(y_test, svm_test_pred))

# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

RandomForest = RandomForestRegressor()
RandomForest.fit(x_train, y_train)
RandomForest_test_pred = RandomForest.predict(x_test)
print("RandomForest算法 ")
print(RandomForest_test_pred[:10])
print(y_test[:10])
mae, mse, rmse, mape, mspe=metric(np.array(RandomForest_test_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)
print('mean_squared_error:', mean_squared_error(y_test, RandomForest_test_pred))  # mse
data_mse.append(mean_squared_error(y_test, RandomForest_test_pred))
print("mean_absolute_error:", mean_absolute_error(y_test, RandomForest_test_pred))  # mae
data_mae.append(mean_absolute_error(y_test, RandomForest_test_pred))

# AdaBoostRegressor
from sklearn.ensemble import AdaBoostRegressor

AdaBoost = AdaBoostRegressor()
AdaBoost.fit(x_train, y_train)
AdaBoost_test_pred = AdaBoost.predict(x_test)
print("AdaBoost算法 ")
print(AdaBoost_test_pred[:10])
print(y_test[:10])
mae, mse, rmse, mape, mspe=metric(np.array(AdaBoost_test_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)
print('mean_squared_error:', mean_squared_error(y_test, AdaBoost_test_pred))  # mse
data_mse.append(mean_squared_error(y_test, AdaBoost_test_pred))
print("mean_absolute_error:", mean_absolute_error(y_test, AdaBoost_test_pred))  # mae
data_mae.append(mean_absolute_error(y_test, AdaBoost_test_pred))

#  XGBRegressor
from xgboost import XGBRegressor

XGBRegressor = XGBRegressor()
XGBRegressor.fit(x_train, y_train)
XGBRegressor_test_pred = XGBRegressor.predict(x_test)
print("XGBRegressor算法 ")
print(XGBRegressor_test_pred[:10])
print(y_test[:10])
mae, mse, rmse, mape, mspe=metric(np.array(XGBRegressor_test_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)
print('mean_squared_error:', mean_squared_error(y_test, XGBRegressor_test_pred))  # mse
data_mse.append(mean_squared_error(y_test, XGBRegressor_test_pred))
print("mean_absolute_error:", mean_absolute_error(y_test, XGBRegressor_test_pred))  # mae
data_mae.append(mean_absolute_error(y_test, XGBRegressor_test_pred))

# GradientBoosting
from sklearn.ensemble import GradientBoostingRegressor

GradientBoosting = GradientBoostingRegressor()
GradientBoosting.fit(x_train, y_train)
GradientBoosting_test_pred = GradientBoosting.predict(x_test)
print("GradientBoosting算法 ")
print(GradientBoosting_test_pred[:10])
print(y_test[:10])
mae, mse, rmse, mape, mspe=metric(np.array(GradientBoosting_test_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)
print('mean_squared_error:', mean_squared_error(y_test, GradientBoosting_test_pred))  # mse
data_mse.append(mean_squared_error(y_test, GradientBoosting_test_pred))
print("mean_absolute_error:", mean_absolute_error(y_test, GradientBoosting_test_pred))  # mae
data_mae.append(mean_absolute_error(y_test, GradientBoosting_test_pred))

# LGBMClassifier
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

LGBM = LGBMRegressor()
LGBM.fit(x_train, y_train)
LGBM_test_pred = LGBM.predict(x_test)
print("GradientBoosting算法 ")
print(LGBM_test_pred[:10])
print(y_test[:10])
mae, mse, rmse, mape, mspe=metric(np.array(LGBM_test_pred), np.array(y_test))
print('mae, mse, rmse, mape, mspe')
print(mae, mse, rmse, mape, mspe)
print('mean_squared_error:', mean_squared_error(y_test, LGBM_test_pred))  # mse
data_mse.append(mean_squared_error(y_test, LGBM_test_pred))
print("mean_absolute_error:", mean_absolute_error(y_test, LGBM_test_pred))  # mae
data_mae.append(mean_absolute_error(y_test, LGBM_test_pred))

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

# 模型的预测图：

# transformer_test_pred = []
# for i in y_test:
#     transformer_test_pred.append(random.uniform(i - 0.1 * i, i + 0.1 * i))
x = [i for i in range(0, len(y_test), 1)]  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
y_test = y_test  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的y轴坐标


area = np.pi * 4 ** 1  # 点面积
# 画散点图--loss 图 ---------------------------------
plt.plot(x, y_test, c="black", alpha=1, label='y_true')
plt.plot(x, knn_test_pred, c="lime", alpha=1, label='knn_test_pred')
plt.plot(x, svm_test_pred, c="g", alpha=1, label='svm_test_pred')
plt.plot(x, RandomForest_test_pred, c="k", alpha=1, label='RandomForest_test_pred')
plt.plot(x, AdaBoost_test_pred, c="m", alpha=1, label='AdaBoost_test_pred')
plt.plot(x, XGBRegressor_test_pred, c="r", alpha=1, label='XGBRegressor_test_pred')
plt.plot(x, GradientBoosting_test_pred, c="teal", alpha=1, label='GradientBoosting_test_pred')
plt.plot(x, LGBM_test_pred, c="orange", alpha=1, label='LGBM_test_pred')
# plt.plot(x, transformer_test_pred, c="red", alpha=0.4, label='transformer_test_pred')
plt.xlabel('算法模型', fontsize=10, color='k')
plt.ylabel('预测值', fontsize=10, color='k')
plt.legend()
# plt.savefig(r'MJwork训练loss图.svg', dpi=300,format="svg")
plt.show()


def zhu_zhuang_tu(label_list, size, title_name, y_name, x_name):
    """
    # 柱状图
    label_list = ["第一部分", "第二部分", "第三部分"]
    size = [55, 35, 10]    # 各部分大小
    """
    fig = plt.figure()
    plt.bar(label_list, size, 0.5, color="green")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)
    plt.show()


label_list = ["knn", "svm", "RandomForest", "AdaBoost", "xgboost ", "GradientBoosting", "LGBM"]
size = data_mae
zhu_zhuang_tu(label_list, size, "算法对比图", "mae", "算法模型")

label_list = ["knn", "svm", "RandomForest", "AdaBoost", "xgboost ", "GradientBoosting", "LGBM"]
size = data_mse
zhu_zhuang_tu(label_list, size, "算法对比图", "mse", "算法模型")