import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('Data.xlsx')
# 2003年-2021年数据
Cities = list(data.loc[:,'城镇化水平'])[::-1]
Birth = list(data.loc[:,'出生率'])[::-1]
GDP = list(map(np.log,list(data.loc[:,'人均国内生产总值'])[::-1]))

GOV = [0 for i in range(len(Birth))]
GOV[-6:] = [1 for i in range(6)] # 2016年起二孩政策出台
# 前几年总和生育率预测
X = np.vstack((Cities,Birth,GDP,GOV)).T
Y = list(data.loc[:,'总和生育率'])[::-1]
Y = np.reshape(Y,(-1,1))
TIME = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
TIME_FUTURE = TIME + [2021+i for i in range(45)]

# 函数拟合结果

# 城镇化水平
def R_U(x):
    return 8.00000000e-01*(1 - (1 + np.exp(-1.59746111e+02 + 7.97438930e-02*x))**-1)
# GDP
def P_GDP(x):
    return 1.14179574e+05*(1+4.16953688e+02*np.exp(-1.64795330e-01*(x-1.97962580e+03)))**-1
# 三孩政策影响因子
tf_2022 = 1.075
error = tf_2022 - (-8.14480597 -11.53925128*R_U(2022) + 1.50804822  *np.log(P_GDP(2022)) + 0.0881199*1)

# 总生育率
def R_TF(t,u):
    if t > 2021:
        return -7.95957099 -13.41615532*R_U(t) + 1.58186331*np.log(P_GDP(t)) + 0.13945147*1 + 0.09*(t-2021)*u*np.exp(0.026*(2021-t))
    elif t > 2016:
        return -7.95957099 -13.41615532*R_U(t) + 1.58186331*np.log(P_GDP(t)) + 0.13945147*1
    else:
        return -7.95957099 -13.41615532*R_U(t) + 1.58186331*np.log(P_GDP(t))
# 二孩理论模型
def FP(x):
    if x < 2016:
        return 0
    else:
        return 1
def TFR(t):
    return 4.755 + 0.5552*FP(t) - 0.348*np.log(P_GDP(t)) - 0.114*R_U(t)

# 绘图
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
fig = plt.figure(figsize=(16,9), dpi=120)
ax = fig.subplots(nrows=1, ncols=2)
ax[0].plot(TIME_FUTURE, R_U(np.asarray(TIME_FUTURE)), color='orange',label='城镇化水平拟合曲线')
ax[0].grid(axis='y', alpha=0.6)
ax[0].scatter(TIME, Cities, c='b', marker='.', alpha=0.8)
ax[0].set_yticks(np.arange(0.4,1,0.05))
ax[0].set_xticks(np.arange(2003,2052,2))
ax[0].set_xticklabels(np.arange(2003,2052,2), rotation=90, fontsize=8)
ax[0].legend()
ax[1].plot(TIME_FUTURE, P_GDP(np.asarray(TIME_FUTURE)), label='人均国内生产总值拟合曲线')
ax[1].grid(axis='y', alpha=0.6)
ax[1].scatter(TIME, np.exp(GDP), c='r', marker='.', alpha=0.6)
ax[1].set_yticks(np.arange(10000,130000,10000))
ax[1].set_xticks(np.arange(2003,2052,2))
ax[1].set_xticklabels(np.arange(2003,2052,2), rotation=90, fontsize=8)
ax[1].legend()

fig2 = plt.figure(figsize=(10,8), dpi=120)
ax2 = fig2.subplots()
ax2.set_title('三孩政策后不同影响程度下的总生育率预测')

for s in [0.8, 0.9, 1.0, 1.1, 1.2]:
    u = s
    u_list = [u for i in range(len(TIME_FUTURE))]
    ax2.plot(TIME_FUTURE, list(map(R_TF, TIME_FUTURE, u_list)), label='影响因子{}'.format(u))

#ax2.plot(TIME_FUTURE, list(map(TFR, TIME_FUTURE)), color='orange', label='理论')
ax2.scatter(TIME, Y, c='b', marker='.', alpha=0.6)
ax2.set_yticks(np.arange(0,2.8,0.2))
ax2.set_xticks(np.arange(2003,2052,2))
ax2.set_xticklabels(np.arange(2003,2052,2), rotation=45, fontsize=8)
ax2.grid(axis='y', alpha=0.6)
ax2.legend()

plt.show()

