import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_excel('Data.xlsx')
# 2003年-2021年数据
Cities = list(data.loc[:,'城镇化水平'])[::-1]
Birth = list(data.loc[:,'出生率'])[::-1]
GDP = list(map(np.log,list(data.loc[:,'人均国内生产总值'])[::-1]))

GOV = [0 for i in range(len(Birth))]
GOV[-6:] = [1 for i in range(6)] # 2016年起二孩政策出台
# 前几年总和生育率预测
X = np.vstack((Cities,GDP,GOV)).T
Y = list(data.loc[:,'总和生育率'])[::-1]
Y = np.reshape(Y,(-1,1))


# 机器学习解
def error(y_true, y_pred):
    return sum((y_true-y_pred)**2)/len(y_pred)
model_L = LinearRegression()
# w = [3, 4, 5, 4, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 6, 3, 2, 1, 0.3]
w = [3, 4, 3, 3, 4, 1, 2, 137, 138, 137, 136, 125, 49, 87, 76, 63, 122, 125, 134]
model_L.fit(X,Y, sample_weight=w)

print('系数：', model_L.coef_)
print('截距：', model_L.intercept_)
print('R2：', model_L.score(X, Y))
k = model_L.coef_[0]
b = model_L.intercept_[0]

f = lambda cities, gdp: b + k[0]*cities + k[1]*gdp

fig = plt.figure(figsize=(8,6), dpi=120)
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Cities, GDP, Y, c='r', marker='o')
Cities, GDP = np.meshgrid(Cities, GDP)
ax.plot_surface(Cities, GDP, f(Cities, GDP), cmap='Greys', cstride=1, rstride=1, alpha=0.5)
ax.set_xlabel('Cities')
ax.set_ylabel('GDP')
ax.set_zlabel('总和生育率')
ax.set_title('线性回归拟合结果')
plt.show()