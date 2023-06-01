import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_excel('Data.xlsx')
# 2003年-2021年数据
Cities = list(data.loc[:,'城镇化水平'])[::-1]
Birth = list(data.loc[:,'出生率'])[::-1]
GDP = list(map(np.log,list(data.loc[:,'人均国内生产总值'])[::-1]))
TIME = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
TIME_FUTURE = TIME + [2021+i for i in range(45)]

# 指数平滑法
def ExpMove(y, a):
    n = len(y)
    M = np.zeros(n)
    M[0] = (y[0]+y[1])/2
    for i in range(1, len(y)):
        M[i] = a*y[i-1]+(1-a)*M[i-1]
    return M

plt.scatter(TIME, Birth, c='r', marker='.', alpha=0.6)
plt.show()
