# import numpy
# #线性代数-----------------------------------------------------------------------------------
# a = numpy.array([[1,2,3], [4,5,6]])
# b = numpy.array([[1,2], [3,4], [5,6]])
# c = numpy.array([[1,2,3],[4,6,6],[7,8,9]])
# d = numpy.array([[9,8,7], [3,2,1]])
# #矩阵加法
# sum = a + d
# #放缩
# e = 3 * a
# #数乘、矩阵乘
# e = numpy.dot(a, b)
# #元素乘
# e = a * d
# #转置
# e = c.T
# #逆矩阵
# result = numpy.linalg.inv(c)
# #行列式
# result = numpy.linalg.det(c)
# #秩
# result = numpy.linalg.matrix_rank(d)
# import matplotlib.pyplot as plt
# #求一次方程组的解
# #矩阵方法
# import time
# import numpy
# start1 = time.time()
# A = numpy.array([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])# 系数矩阵
# b = numpy.array([72, 83, 42])# 常数列(向量)
# x = numpy.linalg.solve(A, b)
# end1 = time.time()
# print(x)
# print(f"耗时{end1-start1:.8f}s")
# #方程方法(可以推广到无穷解的情况)
# from sympy import symbols, Eq, solve
# start2 = time.time()
# x, y, z = symbols('x y z')# 配置字典的键名
# eqs = [Eq(10 * x - y -2 * z, 72),
#        Eq(-x + 10* y - 2 * z, 83),
#        Eq(-x - y + 5 * z, 42)]
# print(solve(eqs, [x, y, z]))
# end2 = time.time()
# x, y = symbols('x y')
# print(f"耗时{end2-start2:.8f}s")
# eqs = [Eq(13 * x - y * 2, 9)]
# print(solve(eqs,[x, y]))
# # 导数---------------------------------------------------------------------
# from scipy.integrate import odeint
# import numpy as np
# import matplotlib.pyplot as plt
# import sympy
# xr = np.arange(-10, 10, 0.1)
# func = lambda x: x**3 + 4*x**2 + 2*x + 500
# x = sympy.symbols('x')
# dfunc = sympy.diff(func(x), x, 1)
# temp_x = []  # 存放导数值的列表
# for i in range(len(xr)):
#     dfunc_x = float(dfunc.evalf(subs={x:xr[i]}))  # 求在x=xr[i]时的导数值
#     temp_x.append(dfunc_x)
# plt.plot(xr, func(xr), 'r-', label='f(x)')
# plt.plot(xr, temp_x, 'b-', label='df/dx')
# plt.grid(True)
# plt.legend()
# plt.show()
# 线性规划----------------------------------------------------------------------------------
# max z = 2*x1 + 3*x2 - 5*x3
# s.t.{x1 + x2 + x3 = 7
#     2*x1 - 5*x2 + x3 >= 10
#     x1 + 3*x2 + x3 <= 12
#     x1, x2, x3 >= 0}

# import scipy
# import numpy as np
# z = np.array([2, 3, -5])  # 标准形式默认求最小值，Ax<=b，若相反则添上负号
# A = np.array([[-2, 5, -1], [1, 3, 1]])
# b = np.array([-10, 12])
# Aeq = np.array([[1, 1, 1]])
# beq = np.array([7])
# x1, x2, x3 = (0, None), (0, None), (0, None)
# res = scipy.optimize.linprog(-z, A, b, Aeq, beq, bounds=(x1, x2, x3))
# print(res)
# 非线性规划-------------------------------------------------------------------------------
# import numpy as np
# import scipy
# def func(x): # 目标函数
#     return 10.5+0.3*x[0]+0.32*x[1]++0.32*x[2]+0.0007*x[0]**2+0.0004*x[1]**2+0.00045*x[2]**2
# cons = ({'type':'eq', 'func':lambda x : x[0]+x[1]+x[2]-700}) # 等式约束
# b1,b2,b3 = (100,200),(120,250),(150,300) # 上下限约束
# x0 = np.array([160,250,290]) # 初始猜测解
# res = scipy.optimize.minimize(func, x0, method='L-BFGS-B', constraints=cons, bounds=(b1,b2,b3))
# print(res)
# # 遗传算法(近似最优解)
# from sko.GA import GA
# import time
# def func(x): # 目标函数
#     return 10.5+0.3*x[0]+0.32*x[1]++0.32*x[2]+0.0007*x[0]**2+0.0004*x[1]**2+0.00045*x[2]**2
# cons = lambda x: x[0]+x[1]+x[2]-700
# start = time.time()
# ga = GA(func=func,n_dim=3,size_pop=500,max_iter=1000,constraint_eq=[cons],lb=[100,120,150],ub=[200,250,300])
# best_x,best_y = ga.run()
# end = time.time()
# print(f"best_x=\n",best_x,"\nbest_y=\n",best_y)
# print(f"耗时{end-start:.8f}s")

# # 指派问题==========================================================================
# # 4名员工完成工作A,B,C,D所需时间如cost矩阵，求最优指派解
# from scipy.optimize import linear_sum_assignment
# import numpy as np
# cost = np.array([[25, 29, 31, 42], [39, 38, 26, 20], [34, 27, 28, 40], [24, 42, 36, 23]])
# row_ind, col_ind = linear_sum_assignment(cost)
# print(row_ind)  #开销矩阵对应的行索引
# print(col_ind)  #开销矩阵对应的列索引
# print(cost[row_ind, col_ind])  #提取每个行索引的最优指派列索引所在的元素，形成数组
# print(cost[row_ind, col_ind].sum())  #数组求和

# # 微分方程求解--------------------------------------------------------------------------

# # 符号解 y"+2y'+y = x^2-------------------------------
# from sympy import *
# y = symbols('y', cls=Function)
# x = symbols('x')
# eq = Eq(y(x).diff(x, 2) + 2*y(x).diff(x,1) + y(x), x*x)
# print(dsolve(eq,y(x)),'\n')
# pprint(dsolve(eq,y(x)))
# # 数值解 y' = x^2 + y^2------------------------------------只能求一阶
# from scipy.integrate import odeint
# from numpy import arange
# dy = lambda y, x: x**2 + y**2
# x = arange(0, 10, 0.5)  # x以0.5的步进从0到10
# sol = odeint(dy, 0 , x)
# print(f"x={x}\n对应数值解为y={sol.T}.")

# # 绘图 y'=1/(x^2+1)-2y^2, y(0)=0----------------------------------------
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt
# from numpy import arange
# dy = lambda y,x: 1/(1+x**2)-2*y**2
# x = arange(0, 10.5, 0.1)
# sol= odeint(dy, 0, x)
# print(f"x={x}\n对应的数值解为y={sol.T}")
# plt.plot(x, sol) # 绘图
# plt.show()

# 高阶微分方程求数值解-------------------------------
# 变量替换，化为一阶微分方程,再用odeint求数值解
# y"-1000(1-y^2)y'+y = 0
# y(0)=0,y'(0)=2
# import matplotlib.pyplot as plt
# from scipy import linspace, exp
# from scipy.integrate import odeint
# import numpy as np
#
# def fvdp(t, y):
#     """
#     要把y看成一个向量，y=[dy0,dy1,dy2,……]分别表示y的n阶导
#     那么y[0]为待解函数，y[1]为一阶导,以此类推……
#     """
#     dy1 = y[1]
#     dy2 = 500*(1-2*y[0]**2)*y[1]-2*y[0]
#     # 注意返回顺序为[一阶导，二阶导]，这样形成一阶微分方程组
#     return [dy1, dy2]
#
# def solve_second_order_ode():
#     """
#     求解二阶ODE
#     """
#     x = np.arange(0.00, 2000, 0.01)
#     y0 = [2, 0]  #初始条件
#     y = odeint(fvdp, y0, x, tfirst=True)
#     # y[:,0]为元组第一个表达式，表示y[x]即为所求，y[:,1]则表示y'(x)
#     y1, = plt.plot(x, y[:,0],"r:", label="y")
#     # 变量名不加','会报错
#     y1_1, = plt.plot(x, y[:,1], label="y'")
#     plt.legend(handles = [y1,y1_1]) #创建图例
#     plt.ylim(-2,2)
#     plt.show()
#
# solve_second_order_ode()

# # 常微分方程组----------------------------
# # dx/dt = 2x - 3y + 3z
# # dy/dt = 4x - 5y + 3z
# # dz/dt = 4x - 4y + 2z
# # x(0)=1, y(0)=2, z(0)=1
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# import numpy as np
# plt.rcParams['font.sans-serif']=['Microsoft YaHei']  #引入微软雅黑字体
#
# def fun1(t, w):
#     x = w[0]
#     y = w[1]
#     z = w[2]
#     return [2*x-3*y+3*z, 4*x-5*y+3*z, 4*x-4*y+2*z]
#
# def fun2(t, w):
#     x = w[0]
#     y = w[1]
#     return [-x**3-y, -y**3+x]
# # 初始条件
# y1 = [1, 2, 1]
#
# y2 = [1, 0.5]
# yy = solve_ivp(fun2, (0,100), y2, method='RK45', t_eval=np.arange(0,100,0.1))
# t = yy.t
# data = yy.y
# plt.plot(t, data[0,:])
# plt.plot(t, data[1,:])
# # plt.plot(t, data[2,:])
# plt.xlabel("时间s")
# plt.show()

# 种群S型增长 r 为增长系数,K为平衡数量，x0为初始数量---------------------案例练习1
# from sympy import *
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint
# from numpy import arange
# x0 = symbols('x0')
# r = symbols('r')
# K = symbols('K')
# t = symbols('t')
# x = symbols('x', cls=Function)
# eq = Eq(x(t).diff(t,1) - r*x(t) + r/K*x(t)**2, 0)
# icss = {x(0): x0}
# print(dsolve(eq, x(t), ics=icss))  # 求方程
# # x(t) = K*x0*exp(r*t)/((-K + x0)*(x0*exp(r*t)/(-K + x0) - 1))
# # 为避免命名冲突，改为g = g(u)
# r = 0.12  # 相关参数设定，增长率r，平衡值K，初始人口g0
# K = 2000000000
# g0 = 140000000
# dg = lambda g, u: r*g - r/K*g**2
# u = arange(0,100,0.5)
# sol = odeint(dg, g0, u)
# print(f"t={u}对应的值为\nx={sol.T}\n")
# plt.plot(u, sol, label='x(t)')
# plt.show()

# 新冠传播模型：SI模型-----------------------案例练习2
# from scipy.integrate import odeint
# import numpy as np
# import matplotlib.pyplot as plt
#
# def dy_dt(y, t, lamda, mu):  # 定义导函数f(y, t)
#     dy_dt = lamda*y*(1-y)
#     return dy_dt

# # 设置模型参数
# number = 1e7  # 总人数
# lamda = 1.0  # 日接触率，患病者每天有效接触的易感者的平均人数
# mu1 = 0.5  # 日治愈率,每天被治愈的患病者人数占患病者总数的比例
# y0 = i0 = 1e-6  # 患病者比例的初值
# tEnd = 50  # 预测日期长度
# t = np.arange(0.0, tEnd, 1)
#
# yAnaly = 1/(1+(1/i0-1)*np.exp(-lamda*t))  # 微分方程的解析解
# yInteg = odeint(dy_dt, y0, t, args=(lamda, mu1))  # 求解微分方程的初值问题
# yDeriv = lamda * yInteg * (1 - yInteg)
#
# # 绘图
# plt.plot(t, yAnaly, '-ob', label='analytic')
# plt.plot(t, yInteg, ':.r', label="numerical")
# plt.plot(t, yDeriv, '-g', label='dy_dt')
# plt.title("Comparison between analytic and numerical solutions")
# plt.legend(loc='right')2
# plt.axis([0, 50, -0.1, 1.1])
# plt.show()

# # 数值方法-----------------------
# # 龙格库塔法
# def runge_kutta(y, x, dx, f):
#     """
#     :param y: is the initial value for y
#     :param x: is the initial value for x
#     :param dx: is the time step in x
#     :param f: is derivative of fuction y(t)
#     :return:
#     """
#     k1 = dx * f(y, x)
#     k2 = dx * f(y + 0.5 * k1, x + 0.5 * dx)
#     k3 = dx * f(y + 0.5 * k2, x + 0.5 * dx)
#     k4 = dx * f(y + k3, x + dx)
#     return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# # 数据与拟合------------------------------------------------------------------------
# # 多项式拟合----------------------------
# import numpy as np
# import scipy.fftpack
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt
# import sympy
# x = np.arange(-1.5, 1.6, 0.5)
# y = [-4.45, -0.45, 0.55, 0.05, -0.44, 0.54, 4.55]
# an = np.polyfit(x, y, 3)  # 用三次多项式拟合
# print(an)  # 打印多项式各项的系数
# p1 = np.poly1d(an)  # 打印多项式
# print(p1)
# # 三种插值------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
# x0 = [1, 2, 3, 4, 5]
# y0 = [1.6, 1.8, 3.2, 5.9, 6.8]
# x = np.arange(1, 5, 1/30)
# f1 = interp1d(x0, y0, 'linear')  # 线性插值
# y1 = f1(x)
# f2 = interp1d(x0, y0, 'cubic')  # 三次样条插值
# y2 = f2(x)
# def lagrange(x0, y0, x):  # 拉格朗日插值
#     y = []
#     for k in range(len(x)):
#         s = 0
#         for i in range(len(y0)):
#             t = y0[i]
#             for j in range(len(y0)):
#                 if i != j:
#                     t *= (x[k]-x0[j])/(x0[i]-x0[j])
#             s += t
#         y.append(s)
#     return y
# y3 = lagrange(x0, y0, x)
# plt.plot(x0, y0, 'r*')
# plt.plot(x,y1,'b',label='linear')
# plt.plot(x,y2,'y',label='cubic')
# plt.plot(x,y3,'r',label='lagrange')
# plt.legend()
# plt.show()
# # 曲线拟合-------------------
# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# X = np.arange(1,11,1)
# Y = [1.1, 2.5, 3.6, 4.9, 6.2, 9.0, 9.5, 11.0, 15.6, 14.1]
# p = np.polyfit(X, Y, 1)
# print(p)
# # 定义一个分式函数去拟合
# def Pfun(X, a, b,c):
#     return 1/(a+b*X)+c
#
# popt, pcov = curve_fit(Pfun, X, Y)  # 使用函数曲线拟合 popt接收计算得到的最佳拟合a，b参数
# print(popt)
#
# plt.plot(X, Y, 'y*')
# plt.plot(X, np.polyval(p, X), 'r-', label='linear_fit')
# plt.plot(X, Pfun(X, *popt), 'b-', label='curve_fit')
# plt.legend()
# plt.show()
# 数据可视化------------------------------------------------------------------------------
# #带限制条件的函数图像-----------------
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.arange(0, 2*np.pi, 0.01)
# y = lambda x: np.sin(x)
# # 添加限制条件x+y<2
# xr=list(x)  # 将x转化为列表，删去列表中不满足限制条件的值
# j = 0
# for i in range(len(xr)):
#     if xr[j]+y(xr[j])<2:
#         xr.pop(j)
#     else:
#         j += 1
# # 绘图
# plt.title('The Graphic of function y=sin(x) While x+y<2')
# plt.plot(xr,y(xr))
# plt.show()
# # 扇形图---------------------
# import matplotlib.pyplot as plt
# labels = ['apple', 'orange', 'banana', 'watermelon']
# sizes = [150, 300, 450, 100]
# explode = (1, 0, 0, 0)
# plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=0)
# plt.axis('equal')
# plt.show()
# # 箱线图------------------
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.random.randint(10, 100, size=(20, 5))  # 生成10-100的5行10列的数
# plt.figure(figsize=(9, 9))  # 画布
# plt.grid(True)  # 显示网格
# plt.boxplot(x,
#             labels=list('ABCDE'),  # 为箱线添加标签
#             sym='g+',  # 异常点形状，默认为蓝色的'+'
#             showmeans=True  # 是否显示均值，默认不显示
#             )
# plt.show()
# # 散点图-----------------------
# import matplotlib.pyplot as plt
# import numpy as np
# x = np.random.randint(10, 100, size=(20, 5))
# y = np.random.randint(10, 100, size=(20, 5))
# plt.scatter(x, y, linewidths=1)
# plt.show()
# # 三维函数图-----------------
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# t = np.arange(0, 10*np.pi, np.pi/50)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(np.sin(t), np.cos(t), t)
# ax.set_xlabel('sin(t)')
# ax.set_ylabel("cos(t)")
# ax.set_zlabel("t")
# plt.show()
# # 三维曲面图---------------
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np
# x1 = np.arange(-5, 5, 0.25)
# y1 = np.arange(-5, 5, 0.25)
# x1, y1 = np.meshgrid(x1, y1)  # 将x，y数据转为二维数据(网格化)
# r1 = np.sqrt(x1**2 + y1**2)
# z1 = np.sin(r1)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # 绘制曲面图
# ax.plot_surface(x1, y1, z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # 设置z轴刻度的范围、位置、格式
# ax.set_zlim(-1, 1)
# plt.show()
# # 等高线图--------------------
# import numpy as np
# import matplotlib.pyplot as plt
# # 计算x,y坐标对应的高度值
# def fun(x, y):
#     return (1-x/2+x**2+y**3)*np.exp(-x**2-y**2)
# # 设置个背景色
# plt.figure()
# n = 256
# x = np.linspace(-3, 3, n)
# y = np.linspace(-3, 3, n)
# # 把x,y数据转化为二维数据(网格化)
# X, Y = np.meshgrid(x, y)
# # 填充等高线
# plt.contourf(X, Y, fun(X,Y))
# plt.show()
# # 可视化的艺术-------------------------------
# import matplotlib.pyplot as plt
# x = [1,2,3,4,5,6,7,8,9]
# y = [0.57,0.35,0.86,0.68,1.34,1.75,2.52,6.43,8.21]
# plt.style.use('ggplot')  # 改变风格 science, ggplot, fivethirtyeight
# plt.figure(figsize=(16, 9), dpi=80)  # 设置图框大小和清晰度
# plt.plot(x, y, 'r^-', linewidth=1)
# plt.xlabel('epoch', size=20)
# plt.ylabel('accurency', size=10)
# plt.legend('y')  # 添加图例
# plt.grid(True)  # 添加网格
# plt.show()
# # 函数与导数的可视化-----------------------
# from scipy.integrate import odeint
# import numpy as np
# import matplotlib.pyplot as plt
# import sympy
#
# xr = np.arange(-10, 10, 0.1)
# func = lambda x: -2**x + 4*x**2 + 2*x + 1
# x = sympy.symbols('x')
# dfunc = sympy.diff(func(x), x, 1)
# temp_x = []  # 存放导数值的列表
# for i in range(len(xr)):
#     dfunc_x = float(dfunc.evalf(subs={x:xr[i]}))  # 求在x=xr[i]时的导数值
#     temp_x.append(dfunc_x)
# plt.plot(xr, func(xr), 'r-', label='f(x)')
# plt.plot(xr, temp_x, 'b-', label='df/dx')
# plt.grid(True)
# plt.legend()
# plt.show()

# # 评价类模型----------------------------------------------------------------------------
# # AHP层次分析法--------------------(主观性强)
# import numpy as np
# A = np.array([[1, 1/3, 1/4, 1/5],
#               [3, 1, 3/4, 3/5],
#               [4, 4/3, 1, 4/5],
#               [5, 5/3, 5/4, 1]])
# m = len(A)  # 获取指标个数(4)
# n = len(A[0])
# RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51]
# R = np.linalg.matrix_rank(A)  # 求判断矩阵的秩
# V, D = np.linalg.eig(A)  # 求判断矩阵的特征值和特征向量, V特征值， D特征向量
# list1 = list(V)
# B = np.max(list1)  # 最大特征值
# index = list1.index(B)
# C = D[:, index]  # 对应特征向量
# CI = (B-n)/(n-1)
# CR = CI/RI[n]
# if CR<0.10:
#     print("CI=", CI)
#     print("CR=", CR)
#     print("对比矩阵A通过一次性检验，各向量权重向量Q为:")
#     sum = np.sum(C)
#     Q = C/sum  # 特征向量标准化
#     print(Q)
# else:
#     print("对比矩阵A未通过一次性检验")
# # TOPSIS分析--------------------------------------
# import pandas as pd
# import numpy as np
# def entropyWeight(data):  # 熵权法
#     data = np.array(data)
#     # 归一化
#     P = data / data.sum(axis=0)
#     # 计算熵值
#     E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
#     # 计算权系数
#     return (1 - E) / (1 - E).sum()
# def topsis(data, weight=None):
#     # 归一化
#     data = data / np.sqrt((data**2).sum())
#     # 最优最劣方案
#     Z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])
#     # 距离
#     weight = entropyWeight(data) if weight is None else np.array(weight)
#     Result = data.copy()
#     Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
#     Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))
#     # 综合得分指数
#     Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
#     Result['排序'] = Result.rank(ascending=False)['综合得分指数']
#     return Result, Z, weight
# # PCA主成分分析-----------------------------------
# import matplotlib.pyplot as plt
# import sklearn.decomposition as dp
# from sklearn.datasets import load_iris
# x, y = load_iris(return_X_y=True)  # 加载数据，x表示数据集中的属性数据，y表示数据标签
# pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
# reduced_x = pca.fit_transform(x, y)  # 对原始数据进行降维，保存在reduced_x中
# red_x, red_y = [], []
# blue_x, blue_y = [], []
# green_x, green_y = [], []
# for i in range(len(reduced_x)):  # 按花的类别将降维后的数据点保存在不同表格中
#     if y[i]==0:
#         red_x.append(reduced_x[i][0])
#         red_y.append(reduced_x[i][1])
#     elif y[i]==1:
#         blue_x.append(reduced_x[i][0])
#         blue_y.append(reduced_x[i][1])
#     else:
#         green_x.append(reduced_x[i][0])
#         green_y.append(reduced_x[i][1])
# plt.scatter(red_x, red_y, c='r', marker='x')
# plt.scatter(blue_x, blue_y, c='b', marker='D')
# plt.scatter(green_x, green_y, c='g', marker='.')
# plt.show()
# # 因子分析------------------------
# import numpy as np
# from factor_analyzer import FactorAnalyzer
# import pandas as pd
# import seaborn as sns
# from matplotlib import pyplot as plt
# from scipy.stats import bartlett
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.style.use('ggplot')
# n_factors = 3  # 因子数量
# cols = ['小学生师比(教师人数=1)', '初中生师比(教师人数=1)', '普通高中生师比(教师人数=1)',
#         '职业高中生师比(教师人数=1)', '普通中专生师比(教师人数=1)', '普通高校生师比(教师人数=1)',
#         '本科院校生师比(教师人数=1)', '专科院校生师比(教师人数=1)']
# # 用检验是否进行
# corr = list(newdata[cols].corr().to_numpy())
# print(bartlett(*corr))
# def kmo(dataset_corr):
#     corr_inv = np.linalg.inv(dataset_corr)
#     nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
#     A = np.ones((nrow_inv_corr, ncol_inv_corr))
#     for i in range(0, nrow_inv_corr):
#         for j in range(0, ncol_inv_corr):
#             A[i,j] = -(corr_inv[i,j])/np.sqrt(corr_inv[i,i]*corr_inv[j,j])
#             A[j,i] = A[i,j]
#     dataset_corr = np.asarray(dataset_corr)
#     kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
#     kmo_denom = kmo_sum + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
#     kmo_value = kmo_num / kmo_denom
#     return kmo_value
# print(kmo(newdata[cols].corr().to_numpy()))
# # 开始计算
# fa = FactorAnalyzer(n_factors=n_factors, method='principal', rotation='varimax')
# fa.fit(newdata[cols])
# communalities = fa.get_communalities()  # 共性因子方差
# loadings = fa.loadings_  # 成分矩阵，可以看出特性的归属因子
# # 画图
# plt.figure(figsize=(9,6), dpi=800)
# ax = sns.heatmap(loadings, annot=True, cmap='BuPu')
# ax.set_xticklabels = (['基础教育', '职业教育', '高等教育'], rotation=0)
# ax.set_yticklabels = (cols, rotation=0)
# plt.title("Factor Analysis")
# plt.savefig("热力图.png", bbox_inches='tight')
# plt.show()
#
# factor_variance = fa.get_factor_variance()  # 贡献率
# fa_score = fa.transform(newdata[cols])  # 因子得分
# # 综合得分
# complex_score = np.zeros([fa_score.shape[0],])
# for i in range(n_factors):
#     complex_score += fa_score[;,i]*factor_variance[1][i]  # 综合得分

# # 图论---------------------------------------------------------------------------------
# # 创建一个图--------------------------------
# import networkx as nx
# import matplotlib.pyplot as plt
# # 创建空的网络
# nf = nx.Graph()
# # 添加节点
# nf.add_node('JFK')
# nf.add_nodes_from(['LAX', 'ATL', 'FLO', 'DFW', 'HNL'])
# print(nf.number_of_nodes())  # 查看节点数,输出7
# # 添加连线(若出现未命名的节点，则自动添加新节点)
# nf.add_edges_from([('JFK','SFD'),('JFK','LAX'),('LAX','ATL'),('FLO','ATL'),
#                    ('ATL','JFK'),('FLO','JFK'),('DFW','HNL')])
# nf.add_edges_from([('OKC','DFW'),('OGG','DFW'),('OGG','LAX')])
# print(nf.number_of_edges())  # 结果为10
# # 绘制网络图
# nx.draw(nf, with_labels=True)
# plt.show()
#
# # print(nx.info(nf)) 报错：不存在info
# print(nx.density(nf))  # 密度
# print(nx.diameter(nf))  # 网络直径
# print(nx.clustering(nf))  # 聚类系数
# print(nx.transitivity(nf))  # 传递性
# print(list(nf.neighbors('OGG')))  # 邻接节点
# print(nx.degree_centrality(nf))  # 度中心性
# print(nx.closeness_centrality(nf))  # 接近中心性
# print(nx.betweenness_centrality(nf))  # 邻接中心性

# # 最短路径与最小生成树----------------------------
# import networkx as nx
# import matplotlib.pyplot as plt
# G = nx.Graph()
# G.add_edge('A', 'B', weight=4)
# G.add_edge('B', 'D', weight=2)
# G.add_edge('A', 'C', weight=3)
# G.add_edge('C', 'D', weight=5)
# G.add_edge('A', 'D', weight=6)
# G.add_edge('C', 'F', weight=7)
# G.add_edge('A', 'G', weight=1)
# G.add_edge('H', 'B', weight=2)
# pos = nx.spring_layout(G)
# # 生成邻接矩阵(报错，不存在to_numpy_matrix)
# # mat = nx.to_numpy_matrix(G)
# # print(mat)
#
# # 最短路径(迪杰斯特拉)
# path = nx.dijkstra_path(G, source='H', target='F')
# print("结点H到F的路径:", path)
# distance = nx.dijkstra_path_length(G, source='H', target='F')
# print('最短路径为:', distance)
#
# # 一点到所有点的最短路径 (报错，target未指定)
# # p = nx.dijkstra_path(G, source='H')
# # d = nx.dijkstra_path_length(G, source='H')
# # for node in G.nodes():
# #     print("H到", node, '的最短路径:', p[node], end='\t')
# #     print("H到", node, '最短路径为:', d[node])
# # 任意两点间的最短距离(貌似不带权值)
# d = nx.shortest_path_length(G)
# d = dict(d)
# for node1 in G.nodes():
#     for node2 in G.nodes():
#         print(node1, "到", node2, "的距离:", d[node1][node2])  # 字典查询
# # 最小生成树
# T = nx.minimum_spanning_tree(G)  # 边有权重
# print(sorted(T.edges(data=True)))
#
# # A*算法最短路径
# p = nx.astar_path(G, source='H', target='F')
# d = nx.astar_path_length(G, source='H', target='F')
# print(p, d)
#
# nx.draw(G, with_labels=True)
# plt.show()
# # 最大流问题--------------------------------------------
# import networkx as nx
# import matplotlib.pyplot as plt
# G = nx.DiGraph()  # 有向图
# G.add_edges_from([('s', 'v1', {'capacity': 8, 'weight': 2}),
#                   ('s', 'v3', {'capacity': 7, 'weight': 8}),
#                   ('v1', 'v3', {'capacity': 5, 'weight': 5}),
#                   ('v1', 'v2', {'capacity': 9, 'weight': 2}),
#                   ('v3', 'v4', {'capacity': 9, 'weight': 3}),
#                   ('v2', 'v3', {'capacity': 2, 'weight': 1}),
#                   ('v4', 't', {'capacity': 10, 'weight': 7}),
#                   ('v2', 't', {'capacity': 5, 'weight': 6}),
#                   ('v4', 'v2', {'capacity': 6, 'weight': 4})])
# pos = nx.spring_layout(G)  # 力导向布局算法默认分配的位置
# pos['t'][0] = 1; pos['t'][1] = 0  # 调节点在图像中的横纵坐标位置
# pos['s'][0] = -1; pos['s'][1] = 0
# pos['v1'][0] = -0.33; pos['v1'][1] = 1
# pos['v3'][0] = -0.33; pos['v3'][1] = -1
# pos['v2'][0] = 0.33; pos['v2'][1] = 1
# pos['v4'][0] = 0.33; pos['v4'][1] = -1
# # pos = nx.spring_layout(G, None, pos={'s':[-1,0], 't':[1,0]}, fixed=['s', 't'])  # 初始化时可选定一些节点的初始位置/固定节点
# # 显示graph
# edge_label1 = nx.get_edge_attributes(G, 'capacity')
# edge_label2 = nx.get_edge_attributes(G, 'weight')
# edge_label = {}
# for i in edge_label1.keys():
#     edge_label[i] = f'({edge_label1[i]:},{edge_label2[i]:})'
# # nx.draw_networkx_edge_labels(G, pos, edge_label, font_size=15)  # 显示原图像
#
# # 处理边上显示的（容量、单位价格）
# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edges(G, pos)
#
# mincostFlow = nx.max_flow_min_cost(G, 's', 't')  # 最小费用最大流
# mincost = nx.cost_of_flow(G, mincostFlow)  # 最小费用的值
# for i in mincostFlow.keys():
#     for j in mincostFlow[i].keys():
#         edge_label[(i, j)] += ',F=' + str(mincostFlow[i][j])
# # 取出每条边流量信息存入边显示值
# nx.draw_networkx_edge_labels(G, pos, edge_label, font_size=12)  # 显示流量及原图
# print(mincostFlow)  # 输出流信息
# print(mincost)
# plt.axis('on')
# plt.xticks([])
# plt.yticks([])
# plt.show()
# # TSP问题(动态规划求解)-------------------
# import numpy as np
# import itertools
# import random
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # TSP问题
# class Solution:
#     def __init__(self, X, start_node):
#         self.X = X  # 距离矩阵
#         self.start_node = start_node  # 开始的结点
#         # 记录处于X结点，未经历M个结点时，矩阵储存X的下一步是M中的哪个结点
#         self.array = [[0]*(2**(len(self.X)-1)) for i in range(len(self.X))]
#     def transfer(self, sets):
#         su = 0
#         for s in sets:
#             su = su + 2**(s-1)  # 二进制转换
#         return su
#     # TSP总接口
#     def tsp(self):
#         s = self.start_node
#         num = len(self.X)
#         cities = list(range(num))  # 形成结点的集合
#         # past_sets = [s]  # 已遍历结点的集合
#         cities.pop(cities.index(s))  # 构建未经历结点的集合
#         node = s  # 初始结点
#         return self.solve(node, cities)  # 求解函数
#     def solve(self, node , future_sets):
#         # 迭代终止条件,表示没有了未遍历结点，直接连接当前结点和起点即可
#         if len(future_sets)==0:
#             return self.X[node][self.start_node]
#         dd = 99999
#         # node如果经过future_sets中结点，最后回到原点的距离
#         ddistance = []
#         # 遍历未经历的结点
#         for i in range(len(future_sets)):
#             s_i = future_sets[i]
#             copy = future_sets[:]
#             copy.pop(i)  # 删除第i个结点，认为已经完成对他的访问
#             ddistance.append(self.X[node][s_i] + self.solve(s_i, copy))
#         # 动态规划递推方程，利用递归
#         dd = min(ddistance)
#         # node需要连接的下一个结点
#         next_one = future_sets[ddistance.index[d]]
#         # 未遍历结点的集合
#         c = self.transfer(future_sets)
#         # 回溯矩阵，（当前节点，未遍历结点集合）--> 下一个结点
#         self.array[node][c] = next_one
#         return d
# # 计算两点间的欧式距离
# def distance(vector1, vector2):
#     d = 0
#     for a, b in zip(vector1, vector2):
#         d += (a-b)**2
#     return d**0.5
# # 随机生成10个坐标点
# n = 10
# random_list = list(itertools.product(range(1, n), range(1, n)))
# cities = random.sample(random_list, n)
# x = []
# y = []
# for city in cities:
#     x.append(city[0])
#     y.append(city[1])
#
# fig = plt.figure()
# plt.scatter(x, y, label='城市位置', s=30)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('TSP问题 随机初始城市')
# plt.legend()
# plt.show()
#
# distance_matrix = np.zeros([n, n])
# for i in range(0, n):
#     for j in range(n):
#         distance1 = distance(cities[i], cities[j])
#         distance_matrix[i][j] = distance1
# S = Solution(distance_matrix, 0)
# print('最短距离:' + str(S.tsp()))
# # 开始回溯
# M = S.array
# lists = list(range(len(S.X)))
# start = S.start_node
# city_order = []
# while len(lists) > 0:
#     lists.pop(lists.index(start))
#     m = S.transfer(lists)
#     next_node = S.array[start][m]
#     print(start, '-->', next_node)
#     city_order.append(cities[start])
#     start = next_node
#
# x1 = []
# y1 = []
# for city in city_order:
#     x1.append(city[0])
#     y1.append(city[1])
#
# x2 = []
# y2 = []
# x2.append(city_order[-1][0])
# x2.append(city_order[0][0])
# y2.append(city_order[-1][1])
# y2.append(city_order[0][1])
#
# plt.plot(x1, y1, label='路线', linewidth=2, marker='o', markersize=8)
# plt.plot(x2, y2, label='路线', linewidth=2, color='r', marker='o', markersize=8)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('TSP问题 路线图')
# plt.legend()
# plt.show()
# # 时间序列-----------------------------------------------------------------------------
# # 移动平均法-----------------
# import numpy as np
# y = np.array([423, 358, 434, 445, 527, 429, 426, 502, 480, 384, 427, 446])
# def MoveAverage(y, N):  # 通过最后N个数据去推测未来数据
#     Mt = [' ']*N
#     for i in range(N+1, len(y)+2):
#         M = y[i-N-1:i-1].mean()
#         Mt.append(round(M, 2))
#     return Mt
# yt3 = MoveAverage(y, 3)
# yt5 = MoveAverage(y, 5)
# s3 = np.sqrt(((y[3:]-yt3[3:-1])**2).mean())
# s5 = np.sqrt(((y[5:]-yt5[5:-1])**2).mean())
# print(yt3)
# print(s3)
# print(yt5)
# print(s5)
# # 指数平滑法----------------------
# import numpy as np
# y = np.array([423, 358, 434, 445, 527, 429, 426, 502, 480, 384, 427, 446])
# def ExpMove(y, a):
#     n = len(y)
#     M = np.zeros(n)
#     M[0] = (y[0]+y[1])/2
#     for i in range(1, len(y)):
#         M[i] = a*y[i-1]+(1-a)*M[i-1]
#     return M
# yt1 = ExpMove(y, 0.2)
# yt2 = ExpMove(y, 0.5)
# yt3 = ExpMove(y, 0.8)
# s1 = np.sqrt(((y-yt1)**2).mean())
# s2 = np.sqrt(((y-yt2)**2).mean())
# s3 = np.sqrt(((y-yt3)**2).mean())
# print(yt1)
# print(yt2)
# print(yt3)
# print(s1)
# print(s2)
# print(s3)
# # 灰色预测-----------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
#
# matplotlib.rcParams['font.sans-serif'] = 'FangSong'
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号‘-’显示为方块的问题
#
#
# class GrayForecast:
#     # 初始化
#     def __init__(self, data, datacolumn=None):
#         if isinstance(data, pd.core.frame.DataFrame):
#             self.data = data
#             try:
#                 self.data.columns = ['数据']
#             except:
#                 if not datacolumn:
#                     raise Exception('您传入的daataframe不止一列')
#                 else:
#                     self.data = pd.DataFrame(data[datacolumn])
#                     self.data.columns = ['数据']
#         elif isinstance(data, pd.core.series.Series):
#             self.data = pd.DataFrame(data, columns=['数据'])
#         else:
#             self.data = pd.DataFrame(data, columns=['数据'])
#
#         self.forecast_list = self.data.copy()
#
#         if datacolumn:
#             self.datacolumn = datacolumn
#         else:
#             self.datacolumn = None
#
#     # 级比检验
#     def level_check(self):
#         # 数据级比检验
#         n = len(self.data)
#         lambda_k = np.zeros(n - 1)
#         for i in range(n - 1):
#             lambda_k[i] = self.data.ix[i]['数据'] / self.data.ix[i + 1]['数据']
#             if lambda_k[i] < np.exp(-2 / (n + 1)) or lambda_k[i] > np.exp(2 / (n + 2)):
#                 flag = True
#         else:
#             flag = False
#         self.lambda_k = lambda_k
#
#         if not flag:
#             print('级比检验失败,请对x(0)作平移变换')
#             return False
#         else:
#             print('级比检验成功，请继续')
#             return True
#
#     # GM(1,1)建模
#     def GM_11_build_model(self, forecast=5):
#         if forecast > len(self.data):
#             raise Exception('您的数据不够')
#         X_0 = np.array(self.forecast_list['数据'].tail(forecast))
#         # 1-AGO
#         X_1 = np.zeros(X_0.shape)
#         for i in range(X_0.shape[0]):
#             X_1[i] = np.sum(X_0[0:i + 1])
#         # 紧邻均值生成序列
#         Z_1 = np.zeros(X_1.shape[0] - 1)
#         for i in range(1, X_1.shape[0]):
#             Z_1[i - 1] = -0.5 * (X_1[i] + X_1[i - 1])
#
#         B = np.append(np.array(np.mat(Z_1).T), np.ones(Z_1.shape).reshape((Z_1.shape[0], 1)), axis=1)
#         Yn = X_0[i:].reshape((X_0[1:].shape[0], 1))
#         B = np.mat[B]
#         Yn = np.mat[Yn]
#         a_ = (B.T * B) ** -1 * B.T * Yn
#         a, b = np.array(a_.T)[0]
#         X_ = np.zeros(X_0.shape[0])
#
#         def f(k):
#             return (X_0[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (k))
#
#         self.forecast_list.loc[len(self.forecast_list)] = f(X_.shape[0])
#
#     # 预测
#     def forecast(self, time=5, forecast_data_len=5):
#         for i in range(time):
#             self.GM_11_build_model(forecast=forecast_data_len)
#
#     # 打印日志
#     def log(self):
#         res = self.forcast_list.copy()
#         if self.datacolumn:
#             res.columns = [self.datacolumn]
#         return res
#
#     # 重置
#     def reset(self):
#         self.forecast_list = self.data.copy()
#
#     # 作图
#     def plot(self):
#         self.forecast_list.plot()
#         if self.datacolumn:
#             plt.ylabel(self.datacolumn)
#             plt.legend([self.datacolumn])
#         plt.show()
#
#
# f = open('电影票房.csv', encoding='utf8')
# df = pd.read_csv(f)
# gf = GrayForecast(df, '票房')
# gf.forecast(10)
# print(gf.log())
# gf.plot()
# # 机器学习与神经网络----------------------------------------------------------------------------------------
# # 机器学习---------------------------
# import numpy as np
# import pandas as pd
# from sklearn.datasets import load_wine #红酒数据集
# from sklearn.metrics import classification_report #存放常用数据指标
# from sklearn.model_selection import train_test_split # 训练集训练集分类器
# import graphviz  # 画文字版决策树
# import pydotplus  # 画图片版决策树
#
# wine = load_wine()
# print(wine.data)  # 数据
# print(wine.target_names)  # 标签名
# print(wine.target)  # 标签值
# print(wine.feature_names)  # 特征名(列名)
# wine_dataframe = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
# print(wine_dataframe)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)  # 训练与测试三七开
#
# from sklearn.linear_model import LogisticRegression  # 逻辑回归
# from sklearn.neighbors import KNeighborsClassifier  # 最近邻回归
# from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯
# from sklearn.tree import DecisionTreeClassifier  # 决策树
# # 集成学习包
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier  # 超越树
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升
#
# clf = DecisionTreeClassifier()  # 创建分类器对象
# clf.fit(Xtrain, Ytrain)  # 训练过程
# Ypredict = clf.predict(Xtest)  # 测试过程
# print(classification_report(Ypredict, Ytest))  # 测试结果
#
# """
# # 妈的为什么会报错，好像文件有问题
# # 只针对决策树
# from sklearn import tree
# tree_data = tree.export_graphviz(
#     clf
#     ,feature_names=wine.feature_names
#     ,class_names=wine.feature_names  # 也可以自己起名
#     ,filled= True  # 填充颜色
#     ,rounded= True  # 决策树边框圆形(方形)
# )
# graph1 = graphviz.Source(tree_data.replace('helvetica', 'Microsoft YaHei UI'), encoding='utf-8')
# graph1.render('./wine_tree')
# """

# # 神经网络------------------------
# from tensorflow import keras
# from tensorflow.keras.datasets import mnist
# from tensorflow.python.keras.utils import np_utils
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers.legacy import RMSprop
#
# # 消除警报
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#
# #数据导入
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_train.shape)
#
# #数据预处理
# x_train = x_train.reshape(x_train.shape) /255.0
# x_test = x_test.reshape(x_test.shape[0], -1) /255.0
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
#
# # 直接使用keras，Sequential搭建全连接网络模型
# model = Sequential()
# model.add(Dense(128, input_shape=(748,)))  """------------------此处会报错,维数不对------------------"""
# model.add(Activation('relu'))
# model.add(Dense(10))
# model.add(Activation('softmax'))
#
# # lr为学习率，epsilon防止出现0， rho/decay分别对应公式中beta_1和beta_2
# rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.00001)
# model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
# print('-----------------------training-----------------------')
# model.fit(x_train, y_train, epochs=5, batch_size=32)
# print('\n')
# print('-----------------------testing-------------------------')
# loss, accuracy = model.evaluate(x_test, y_test)
# print('loss:',loss)
# print('accuracy:',accuracy)

# # 遗传算法--------------------------------------------
# import numpy as np
#
# def schaffer(p):
#     x1, x2 = p
#     x = np.square(x1)+ np.square(x2)
#     return 0.5 + (np.square(np.sin(x))-0.5) / np.square(1+0.001*x)
#
# from sko.GA import GA
# ga = GA(func=schaffer, n_dim=2, size_pop=50, max_iter=800, prob_mut=0.001, lb=[-1,-1], ub=[1,1], precision=1e-7)
# best_x, best_y = ga.run()
# print('best_x:',best_x,'\nbest_y:',best_y)
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# Y_history = pd.DataFrame(ga.all_history_Y)
# fig, ax = plt.subplots(2,1)
# ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
# Y_history.min(axis=1).cummin().plot(kind='line')
# plt.show()
# # 遗传算法2-----------------------------------------------
# import numpy as np
# from scipy import spatial
# import matplotlib.pyplot as plt
#
# num_points = 50
# points_coordinate = np.random.rand(num_points, 2)
# distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
#
# def cal_total_distance(routine):
#     num_points, = routine.shape
#     return sum([distance_matrix[routine[i%num_points], routine[(i+1)%num_points]] for i in range(num_points)])
#
# from sko.GA import GA_TSP
#
# ga_TSP = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
# best_points, best_distance = ga_TSP.run()
#
# fig, ax = plt.subplots(1,2)
# best_points_ = np.concatenate([best_points,[best_points[0]]])
# best_points_coordinate = points_coordinate[best_points_,:]
# ax[0].plot(best_points_coordinate[:,0],best_points_coordinate[:,1], 'o-r')
# ax[1].plot(ga_TSP.generation_best_Y)
# plt.show()

# # 蚁群算法----------------------------------------
# # 粒子群算法------------------------------------
# import numpy as np
# from sko.PSO import PSO
#
# def demo_func(x):
#     x1,x2,x3=x
#     return x1**2 + (x2-0.05)**2 + x3**2
# pso = PSO(func=demo_func, dim=3, pop=40, max_iter=150, lb=[0,-1,0.5], ub=[1,1,1], w=0.8, c1=0.5, c2=0.5)
# pso.run()
# print("best_X is:", pso.gbest_x, '\nbest_Y is:', pso.gbest_y)
#
# import matplotlib.pyplot as plt
# plt.plot(pso.gbest_y_hist)
# plt.show()

# # 回归分析-----------------------------------------------------------------------------
# sklearn回归模型------------
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import leastsq
# 定义自变量 X 和因变量 y
X = np.random.randn(100, 5)
y = np.dot(X, [0.5, 2.0, -1.5, 3.0, 0.2]) + 0.5 * np.random.randn(100)

# 创建一个线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, y)

# 输出模型参数
print('系数：', model.coef_)
print('截距：', model.intercept_)
print('R2：', model.score(X, y))

# 最小二乘回归模型--------------
def model(params, x):
    return np.dot(x, params[:-1]) + params[-1]

# 误差函数
def error(params, x, y):
    return np.hstack((model(params, x) - y,params[-1]))

# 初始化参数
params0 = np.zeros(6)

# 最小二乘法求解
params_fit, success = leastsq(error, params0, args=(X, y))

# 打印结果
print(params_fit)
print(success)





