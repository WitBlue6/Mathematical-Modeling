import numpy as np
import pandas as pd
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = pd.read_excel('Data.xlsx')
# 2003年-2021年数据
Cities = list(data.loc[:,'城镇化水平'])[::-1]
Birth = list(data.loc[:,'出生率'])[::-1]
GDP = list(data.loc[:,'人均国内生产总值'])[::-1]
TIME = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
GOV = [0 for i in range(len(Birth))]
GOV[-6:] = [1 for i in range(6)] # 2016年起二孩政策出台
# 前几年总和生育率预测
X = np.vstack((Cities,Birth,GDP,GOV)).T
Y = list(data.loc[:,'总和生育率'])[::-1]
Y = np.reshape(Y,(-1,1))

# 拟合函数
def CITIES_TIME(x,a,d,e):
        return a*(1-(1 + np.exp(e*x+d))**-1)


popt, pcov = curve_fit(CITIES_TIME, TIME, Cities, bounds=([0.7,-200,0],[0.8,0,0.1]),maxfev=50000000)
plt.plot(TIME, CITIES_TIME(np.asarray(TIME), *popt), label='fit_curve')
plt.scatter(TIME, Cities, c='r', marker='o', alpha=0.6)
plt.show()
print(popt)