# 微分方程-----------------------------------------------------------------------

# # 解析解---------
# import sympy
# import numpy as np
# from scipy.integrate import odeint
# def apply_ics(sol, ics, x, known_params):
#     free_params = sol.free_symbols - set(known_params)
#     eqs = [(sol.lhs.diff(x,n) - sol.rhs.diff(x,n)).subs(x,0).subs(ics) for n in range(len(ics))]
#     sol_params = sympy.solve(eqs, free_params)
#     return sol.subs(sol_params)
#
# sympy.init_printing()  # 初始化打印环境
# t, omega0, gamma = sympy.symbols("t, omega_0, gamma", positive=True)  # 标记参数，且正数
# x = sympy.Function('x')  # 标记x是微分函数，非变量
# ode = x(t).diff(t, 2)+2*gamma*omega0*x(t).diff(t, 1)+omega0**2*x(t)
# ode_sol = sympy.dsolve(ode)  # 用diff()和dsolve求解
# ics = {x(0): 1, x(t).diff(t).subs(t, 0): 0}  # 初始条件与字典匹配
# x_t_sol = apply_ics(ode_sol, ics, t, [omega0, gamma])
# sympy.pprint(x_t_sol)

# # 数值解---------------
# import numpy as np
# from scipy import integrate
# import matplotlib.pyplot as plt
# import sympy
# def plot_direction_field(x, y_x, f_xy, x_lim=(-5,5), y_lim=(-5,5), ax=None):
#     f_np = sympy.lambdify((x, y_x), f_xy, 'numpy')  # 转成numpy矩阵(算得快)
#     x_vec = np.linspace(x_lim[0], x_lim[1], 20)
#     y_vec = np.linspace(y_lim[0], y_lim[1], 20)
#     if ax is None:
#         _, ax = plt.subplots(figsize=(4, 4))
#     dx = x_vec[1] - x_vec[0]
#     dy = y_vec[1] - x_vec[0]
#     for m,xx in enumerate(x_vec):
#         for n,yy in enumerate(y_vec):
#             Dy = f_np(xx, yy)*dx
#             Dx = 0.8*dx**2/np.sqrt(dx**2 + Dy**2)
#             Dy = 0.8*Dy*dy/np.sqrt(dx**2 + Dy**2)
#             ax.plot([xx-Dx/2, xx+Dx/2], [yy-Dy/2, yy+Dy/2], 'b', lw=0.5)
#     ax.axis('tight')
#     ax.set_title(r"$%s$"%(sympy.latex(sympy.Eq(y_x.diff(x), f_xy))), fontsize=18)
#     return ax
# x = sympy.symbols('x')
# y = sympy.Function('y')
# f = x-y(x)**2
# f_np = sympy.lambdify((y(x), x), f)  # 用符号表达式转隐函数
# y0 = 1
# xp = np.linspace(0, 5, 100)
# yp = integrate.odeint(f_np, y0, xp)  # 初始y0解f_np, x范围xp
# xn = np.linspace(0, -5, 100)
# yn = integrate.odeint(f_np, y0, xp)
# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# plot_direction_field(x, y(x), f, ax=ax)  # 绘制f的场线图
# ax.plot(xn, yn, 'b', lw=2)
# ax.plot(xp, yp, 'r', lw=2)
# plt.show()

# from scipy.optimize import minimize
# import math
# import numpy as np
# def fun(x):
#     return x[0] * math.sqrt((x[1] - 1.25) ** 2 + (x[2] - 1.25) ** 2) + x[3] * math.sqrt((x[14] - 1.25) ** 2 + (x[15] - 1.25) ** 2) + x[4] * math.sqrt((x[1] - 8.75) ** 2 + (x[2] - 0.75) ** 2) + x[5] * math.sqrt((x[14] - 8.75) ** 2 + (x[15] - 0.75) ** 2) + x[6] * math.sqrt((x[1] - 0.5) ** 2 + (x[2] - 4.75) ** 2) + x[7] * math.sqrt((x[14] - 0.5) ** 2 + (x[15] - 4.75) ** 2) + x[8] * math.sqrt((x[1] - 5.75) ** 2 + (x[2] - 5) ** 2) + x[9] * math.sqrt((x[14] - 5.75) ** 2 + (x[15] - 5) ** 2) + x[10] * math.sqrt((x[1] - 3) ** 2 + (x[2] - 6.5) ** 2) + x[11] * math.sqrt((x[14] - 3) ** 2 + (x[15] - 6.5) ** 2) + x[12] * math.sqrt((x[1] - 7.25) ** 2 + (x[2] - 7.25) ** 2) + x[13] * math.sqrt((x[14] - 7.25) ** 2 + (x[15] - 7.25) ** 2)
# cons1 = {'type': 'ineq', 'fun': lambda x: x[0] + x[3] - 3}
# cons2 = {'type': 'ineq', 'fun': lambda x: x[4] + x[5] - 5}
# cons3 = {'type': 'ineq', 'fun': lambda x: x[6] + x[7] - 4}
# cons4 = {'type': 'ineq', 'fun': lambda x: x[8] + x[9] - 7}
# cons5 = {'type': 'ineq', 'fun': lambda x: x[10] + x[11] - 6}
# cons6 = {'type': 'ineq', 'fun': lambda x: x[12] + x[13] - 11}
# # 以上为限制条件
# cons7 = {'type': 'ineq', 'fun': lambda x: -(x[0] + x[4] + x[6] + x[8] + x[10] + x[12] - 20)}
# cons8 = {'type': 'ineq', 'fun': lambda x: -(x[3] + x[5] + x[7] + x[9] + x[11] + x[13] - 20)}
# # 以上为限制条件2
# x0 = np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
#                1, 1, 1, 1, 1, 1])
# b,a = (0, 11),(0, 20)
# bnds = (b, a, a, b, b, b, b, b, b, b, b, b, b, b, a, a)
# cons = (cons1, cons2, cons3, cons4, cons5, cons6, cons7, cons8)
# res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
# print(res)
#
# import requests
# import bs4
# headers = {
# "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0"}
# res = requests.get("https://movie.douban.com/top250", headers=headers)
# soup = bs4.BeautifulSoup(res.text, 'html.parser')
# targets = soup.find_all('div', class_='hd')
# for each in targets:
#     print(each.a.span.text)
# # 泰勒展开可视化模拟
# import sympy
# import matplotlib.pyplot as plt
# import numpy as np
# x = sympy.symbols('x')
# f = lambda x: 5*sympy.sin(2*x)-sympy.exp(x)
# print(sympy.nsolve(f(x), 2))
# print(sympy.nsolve(f(x).diff(x), 2))
# print(sympy.limit(f(x).diff(x,2), x, 0).removeO())
# print(f(x).series(x,0,15))
# xr = np.linspace(-5,5,100)
# ff = lambda x: 5*np.sin(2*x)-np.exp(x)
# taylor5_f = lambda x: -1 + 9*x - x**2/2 - 41*x**3/6 - x**4/24
# taylor10_f = lambda x: -1 + 9*x - x**2/2 - 41*x**3/6 - x**4/24 + 53*x**5/40 - x**6/720 - 641*x**7/5040 - x**8/40320 + 853*x**9/120960
# taylor15_f = lambda x: -1 + 9*x - x**2/2 - 41*x**3/6 - x**4/24 + 53*x**5/40 - x**6/720 - 641*x**7/5040 - x**8/40320 + 853*x**9/120960 - x**10/3628800 - 133*x**11/518400 - x**12/479001600 + 1517*x**13/230630400 - x**14/87178291200
# plt.figure()
# plt.plot(xr, ff(xr), 'k', label='y=f(x)')
# plt.plot(xr, taylor5_f(xr), 'r', label='taylor expansion 5')
# plt.plot(xr, taylor10_f(xr), 'b', label='taylor expansion 10')
# plt.plot(xr, taylor15_f(xr), 'g', label='taylor expansion 15')
# plt.title('Taylor Expansion at x=0 of different series of\nFuction: y=5sin(2x)-e^x')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.ylim(-100,100)
# plt.legend()
# plt.grid()
# plt.show()

# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = 'FangSong'
# plt.figure(figsize=(20, 8), dpi=80)
# x = ['阿凡达2','流浪地球2','满江红','屌丝之倪天宇的逆袭']
# y1 = [53,35,66,24]
# y2 = [57,77,79,25]
# x1 = list(range(len(x)))
# x2 = [i+0.2 for i in x1]
# x3 = [i+0.2*2 for i in x1]
# plt.bar(x1, y1, width=0.2, label='9月')
# plt.bar(x2, y2, width=0.2, label='10月')
# plt.bar(x3, y2, width=0.2, label='11月')
# plt.xticks(x2, x)
# plt.grid(alpha=0.4)
# plt.xlabel('电影')
# plt.ylabel('票房(万)')
# plt.title('时间段内电影票房图')
# plt.legend()
# plt.savefig('./movie.png')
# plt.show()

# import numpy as np
# a = np.arange(10).reshape(2,5)
# print(a)
# print(a.sum(axis=1))  # 行求和
# print(a.sum(axis=0))  # 列求和
# # 布尔索引
# print(a<5)
# a[a<5] = 0
# print(a)
# # 三元运算符
# np.where(a<5,0,10)  # x = 0 if x<5 else 10
# # 裁剪
# a.clip(3,7) # x=3 if x<3 , x=7 if x>7
# # 拼接
# t1 = np.arange(10).reshape(2,5)
# t2 = np.arange(10,20).reshape(2,5)
# print('*'*100)
# print(np.vstack((t1,t2)))  #竖直拼接
# print(np.hstack((t1,t2)))  #水平拼接
# # 行列交换
# a[[0,1],:]=a[[1,0],:]  # 行交换
# a[:,[0,1]]=a[:,[1,0]]  # 列交换

# import pandas as pd
# import numpy as np
# import string
# #t = pd.Series(np.arange(10), index=list(string.ascii_uppercase[:10]))
# a = {string.ascii_uppercase[i]:i for i in range(10)}
# t = pd.Series(a)
# t=t.astype(float)
# print(t)
# print(t['B'])
# print(t[1])

# import pandas as pd
# import numpy as np
# d1 = {'name':['xiaoming','xiaogang'], 'age':[19,14], 'tel':[10086,10001]}
# d2 = [{'name':'xiaohong', 'age':'17', 'tel':'10086'}, {"name":'xiaohong', "age":"18"}]
# t = pd.DataFrame(np.arange(12).reshape(3,4), index=list('xyz'), columns=list('abcd'))
# print(t.loc['x','c'])  # 标签索引
# print(t.loc[:,'c'])
# print(t.loc['x':'z',['a','b']])
# print(t[:2])  # 好像不能列索引
# print(t.iloc[:2,:1])  # 下标索引
# t[t['b']<20] = 100  # 布尔索引(用于列索引)
# print(t['b'])
#print(pd.DataFrame(d1))
#print(pd.DataFrame(d2))
# a = pd.DataFrame(d1)
#print(a.info()).
#print(a.describe())
# a = a.sort_values(by='age', ascending=True) #升序
# print(a.head(1))
# print('*'*100)
# print(a.tail(1))  # 显示最后一个
# 导演的人数
# print(len(set(t['Director'].tolist())))
# print(len(t['Director'].unique()))
# 获取演员的人数
# actors_list = t['Actors'].str.split(',').tolist()
# actors = [j for i in actors_list for j in i]

# import numpy as np
# import pandas as pd
# df = pd.DataFrame(np.arange(9).reshape(3,3),columns=list('abc'))
# grouped = df.groupby('b')
# for i,j in grouped:
#     print(i)
#     print('-'*100)
#     print(j)
#     print('*'*100)
#
# print(grouped['a'].count())

# import pandas as pd
# import numpy as np
# #index = pd.date_range(start='20171230', end='20200301',freq='D')
# #print(df)
# index = pd.date_range(start='20171230', periods=10,freq='H')
# #print(df)
# df = pd.DataFrame(np.random.rand(100).reshape(10,10), index=index)
# print(df)

# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=(10,8), dpi=80)
# x = np.arange(-2,2,0.001)
# y = np.arange(-2,2,0.001)
# x,y = np.meshgrid(x,y)
# z = lambda x,y:np.arctan((np.abs(x)+np.abs(y))/(x**2+y**2))
# z = lambda x,y: np.log((3*x**2-x**2*y**2+3*y**2)/(x**2+y**2))
# ax = fig.add_subplot(111,projection='3d')
# ax.plot_surface(x,y,z(x,y),cmap='rainbow',alpha=0.8)
# fig.tight_layout()
# plt.show()
# import numpy as np
# from sko.GA import GA_TSP
#
# # 城市坐标
# city_locations = np.array([[1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556], [3238, 1229], [4196, 1004], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756], [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2367], [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826], [2370, 2975]])
# num_cities = len(city_locations)
#
# # 生成距离矩阵
# distance_matrix = np.zeros((num_cities, num_cities))
# for i in range(num_cities):
#     for j in range(i, num_cities):
#         distance_matrix[i][j] = np.sqrt(np.sum((city_locations[i] - city_locations[j]) ** 2))
#         distance_matrix[j][i] = distance_matrix[i][j]
#
# # 定义目标函数，即路径总长度
# def cal_total_distance(routine):
#     num_points, = routine.shape
#     return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
#
# # 定义遗传算法模型
# ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_cities, size_pop=100, max_iter=500, prob_mut=0.01)
#
# # 运行遗传算法
# best_route, best_distance = ga_tsp.run()
#
# # 输出结果
# print('最优路径:', best_route)
# print('最优路径长度:', best_distance)

# import pygame
# import random
#
# # 游戏面板大小
# GRID_SIZE = 4
#
# # 块大小
# TILE_SIZE = 80
# MARGIN = 20
#
# # 颜色
# BLACK = (0, 0, 0)
# WHITE = (255, 255, 255)
# GRAY = (128, 128, 128)
# RED = (255, 0, 0)
#
#
# # 初始化游戏面板
# def initialize_board():
#     board = [[0 for i in range(GRID_SIZE)] for j in range(GRID_SIZE)]
#     add_tile(board)
#     add_tile(board)
#     return board
#
#
# # 在游戏面板上添加新块
# def add_tile(board):
#     empty_cells = []
#     for i in range(GRID_SIZE):
#         for j in range(GRID_SIZE):
#             if board[i][j] == 0:
#                 empty_cells.append((i, j))
#     if empty_cells:
#         i, j = random.choice(empty_cells)
#         board[i][j] = 2
#
#
# # 判断游戏是否结束
# def is_game_over(board):
#     for i in range(GRID_SIZE):
#         for j in range(GRID_SIZE):
#             if board[i][j] == 0:
#                 return False
#             if i < GRID_SIZE - 1 and board[i][j] == board[i + 1][j]:
#                 return False
#             if j < GRID_SIZE - 1 and board[i][j] == board[i][j + 1]:
#                 return False
#     return True
#
#
# # 在游戏面板上移动块
# def move_tiles(board, direction):
#     if direction == "up":
#         for j in range(GRID_SIZE):
#             for i in range(1, GRID_SIZE):
#                 if board[i][j] != 0:
#                     for k in range(i, 0, -1):
#                         if board[k - 1][j] == 0:
#                             board[k - 1][j] = board[k][j]
#                             board[k][j] = 0
#                         elif board[k - 1][j] == board[k][j]:
#                             board[k - 1][j] *= 2
#                             board[k][j] = 0
#                             break
#     elif direction == "down":
#         for j in range(GRID_SIZE):
#             for i in range(GRID_SIZE - 2, -1, -1):
#                 if board[i][j] != 0:
#                     for k in range(i, GRID_SIZE - 1):
#                         if board[k + 1][j] == 0:
#                             board[k + 1][j] = board[k][j]
#                             board[k][j] = 0
#                         elif board[k + 1][j] == board[k][j]:
#                             board[k + 1][j] *= 2
#                             board[k][j] = 0
#                             break
#     elif direction == "left":
#         for i in range(GRID_SIZE):
#             for j in range(1, GRID_SIZE):
#                 if board[i][j] != 0:
#                     for k in range(j, 0, -1):
#                         if board[i][k - 1] == 0:
#                             board[i][k - 1] = board[i][k]
#                             board[i][k] = 0
#                         elif board[i][k - 1] == board[i][k]:
#                             board[i][k - 1] *= 2
#                             board[i][k] = 0
#                             break
#
# # 绘制游戏面板
# def draw_board(board, screen):
#     screen.fill(GRAY)
#     font = pygame.font.Font(None, 60)
#     for i in range(GRID_SIZE):
#         for j in range(GRID_SIZE):
#             tile = board[i][j]
#             rect = pygame.Rect(j * (TILE_SIZE + MARGIN) + MARGIN, i * (TILE_SIZE + MARGIN) + MARGIN, TILE_SIZE,
#                                TILE_SIZE)
#             pygame.draw.rect(screen, WHITE, rect)
#             if tile != 0:
#                 text = font.render(str(tile), True, BLACK)
#                 text_rect = text.get_rect()
#                 text_rect.center = rect.center
#                 screen.blit(text, text_rect)
#
#
# # 主函数
# def main():
#     pygame.init()
#
#     # 设置窗口
#     screen_width = GRID_SIZE * (TILE_SIZE + MARGIN) + MARGIN
#     screen_height = GRID_SIZE * (TILE_SIZE + MARGIN) + MARGIN
#     screen = pygame.display.set_mode((screen_width, screen_height))
#     pygame.display.set_caption("2048")
#
#     # 初始化游戏面板
#     board = initialize_board()
#
#     # 游戏主循环
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 return
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_UP:
#                     move_tiles(board, "up")
#                     add_tile(board)
#                 elif event.key == pygame.K_DOWN:
#                     move_tiles(board, "down")
#                     add_tile(board)
#                 elif event.key == pygame.K_LEFT:
#                     move_tiles(board, "left")
#                     add_tile(board)
#                 elif event.key == pygame.K_RIGHT:
#                     move_tiles(board, "right")
#                     add_tile(board)
#
#         # 绘制游戏面板
#         draw_board(board, screen)
#         pygame.display.update()
#
#         # 判断游戏是否结束
#         if is_game_over(board):
#             font = pygame.font.Font(None, 100)
#             text = font.render("Game Over!", True, RED)
#             text_rect = text.get_rect()
#             text_rect.center = screen.get_rect().center
#             screen.blit(text, text_rect)
#             pygame.display.update()
#             pygame.time.wait(3000)
#             return
#
#
# if __name__ == "__main__":
#     main()

# from scipy import integrate
# import numpy as np
# import matplotlib.pyplot as plt
#
# xr = np.arange(0,5,0.01)
# y = lambda x: np.exp(x)*np.sin(2*x) + x**3 - 4*x**2
# plt.figure(figsize=(10,8), dpi=80)
# plt.axis([np.min(xr),np.max(xr),np.min(y(xr))-4,np.max(y(xr))+4])
# plt.plot(xr,y(xr), 'r', label='y=sin2x + lnx + x^3 - 4x^2')
# plt.plot(xr,y(xr)-y(xr),'k-')
# plt.legend()
# plt.fill_between(xr, y1=y(xr), y2=0, where=(xr>=1)&(xr<=4), facecolor='blue', alpha=0.2)
#
# fArea, err = integrate.quad(y,1,4)
# plt.text(0.5, 30, f'quad={fArea}', fontsize=18)
# plt.show()


# # lnI-U图像拟合
# import matplotlib.pyplot as plt
# import numpy as np
#
# U = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# I = [0.0108, 0.0270, 0.0530, 0.0944, 0.16201, 0.283, 0.484, 0.837, 1.522, 3.1]
# lnI = list(map(np.log,I))
#
# plt.figure(figsize=(12,9), dpi=120)
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
#
# yticks = np.linspace(-4.600,1.200,30)
# yticks_str = ['{:.3f}'.format(i) for i in yticks]
# plt.yticks(yticks, yticks_str)  # 设置格式化表示
# plt.xticks([i for i in U])
# plt.xlabel("U/V")
# plt.ylabel("lnI/mA")
# plt.title('lnI-U拟合图像\n')
#
# plt.scatter(U, np.log(I))  # 散点图
# for i in range(len(U)):
#     plt.text(U[i]-0.2, lnI[i]-0.2, f"({U[i]},{lnI[i]:.3f})")  # 绘制坐标信息
# p = np.polyfit(U, lnI, 1)
# print(p)  # 拟合直线的系数
# plt.plot(U, np.polyval(p, U), 'r')  # 拟合曲线
#
# plt.show()

# # P-R图像
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit
# # 原始数据
# R = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000, 3000, 4000, 10000]
# U = [0.11, 0.65, 1.25, 1.83, 2.37, 2.98, 3.39, 3.79, 4.07, 4.30, 4.46, 4.59, 4.68, 4.75, 4.80, 4.85, 5.00, 5.12, 5.17, 5.26]
# I = [6.04, 6.0, 5.9, 5.8, 5.8, 5.7, 5.5, 5.3, 5.0, 4.7, 4.4, 4.1, 3.9, 3.6, 3.4, 3.2, 2.5, 1.7, 1.3, 0.522]
# P = []
# for i in range(len(U)):
#     P.append(U[i]*I[i])
# P = [1.45, 5.152, 7.504, 9.912, 12.925, 15.336, 17.543, 18.787, 18.912, 18.81, 18.354, 17.55, 16.946,  16.275, 15.2, 14.756, 11.362, 8.08, 6.12, 2.59]
# # 图像基本设置
# plt.figure(figsize=(12,9), dpi=120)
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.xticks(np.linspace(R[0],R[-1],21))
# plt.yticks(np.linspace(0,30,11),['{:.3f}'.format(i) for i in np.linspace(0,30,11)])
# plt.xlabel('R/Ω')
# plt.ylabel('P/mW')
# plt.title('P-R拟合图像\n')
#
# # 标记点
# for i in range(len(P)):
#     if(i <= 6 or i >= 11):
#         plt.text(R[i], P[i], '{:.3f}'.format(P[i]))
# #plt.text(800+50, 20.4, '{:.3f}'.format(4.07*5.0))
# plt.scatter(R, P, c='r', marker='^')

# #多项式拟合
# r = np.polyfit(R, P, 12)
# plt.plot(R, np.polyval(r, R), 'k')
#
# # #曲线拟合
# # def Pfun(X, a, b, c):
# #     return a*X/(c*X+b)**2
# # popt, pcov = curve_fit(Pfun, R, P, maxfev=50000)
# # plt.plot(R, Pfun(np.asarray(R), *popt))
#
# plt.show()

# # I-U图像
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit
# # 原始数据
# U = [0.11, 0.65, 1.25, 1.83, 2.37, 2.98, 3.39, 3.79, 4.07, 4.30, 4.46, 4.59, 4.68, 4.75, 4.80, 4.85, 5.00, 5.12, 5.17, 5.26]
# I = [6.04, 6.0, 5.9, 5.8, 5.8, 5.7, 5.5, 5.3, 5.0, 4.7, 4.4, 4.1, 3.9, 3.6, 3.4, 3.2, 2.5, 1.7, 1.3, 0.522]
#
# # 图像基本设置
# plt.figure(figsize=(12,9), dpi=120)
# plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
# plt.xticks(np.linspace(0,6,7))
# plt.yticks(np.linspace(0,7,8),['{:.3f}'.format(i) for i in np.linspace(0,7,8)])
# plt.xlabel('U/V')
# plt.ylabel('I/mA')
# plt.title('I-U拟合图像\n刘作豪 1120222314')
#
# # 标记点
# for i in range(len(U)):
#     plt.text(U[i], I[i], '{:.3f}'.format(I[i]))
# plt.scatter(U, I, c='r', marker='.')
#
# # 使用指数函数拟合
# def Pfun(X, a, b, c):
#     return -a*np.exp(c*X)+b
# popt, prov = curve_fit(Pfun, U, I, maxfev=5000000)
# plt.plot(U, Pfun(np.asarray(U), *popt))
#
# plt.show()

# from DataFit import Fit
# import numpy as np
# import random
# X = np.linspace(1, 100, 100)
# Y = [np.log(X[i]) + random.random()/10 for i in range(len(X))]
# fit = Fit.DataFit(X,Y)
# fit.scatter()
# data = fit.DataInfo()
# def Pfun(X, a, b):
#     return np.log(a*X) + b
# fit.curve_fit(Pfun)
# fit.polyfit(6)
# fit.showfig()


# import wordcloud
# w = wordcloud.WordCloud(width=600, height=400, font_path='msyh.ttc')
# w.min_font_size = 20
# w.max_font_size = 150
# w.background_color = 'white'
# txt = '''
# Beautiful is better than ugly.
# Explicit is better than implicit.
# Simple is better than complex.
# Complex is better than complicated.
# Flat is better than nested.
# Sparse is better than dense.
# Readability counts.
# Special cases aren't special enough to break the rules.
# Although practicality beats purity.
# Errors should never pass silently.
# Unless explicitly silenced.
# In the face of ambiguity, refuse the temptation to guess.
# There should be one-- and preferably only one --obvious way to do it.
# Although that way may not be obvious at first unless you're Dutch.
# Now is better than never.
# Although never is often better than *right* now.
# If the implementation is hard to explain, it's a bad idea.
# If the implementation is easy to explain, it may be a good idea.
# Namespaces are one honking great idea -- let's do more of those!
# '''
# w.generate(txt)
# w.to_file('The Zen of Python.png')

# # 检索陈后腰
# import wordcloud
# import jieba
# import pandas as pd
# jieba.setLogLevel(jieba.logging.INFO)
# f = open('chenhouyao.txt', 'r', encoding='utf-8')
# words = f.readlines()
# f.close()
# to_be_removed = []
# for i in words:
#     if i.find('QQ') != -1:
#         to_be_removed.append(i)
#     elif i.find('陈厚尧') != -1:
#         to_be_removed.append(i)
#     elif i == '\n':
#         to_be_removed.append(i)
# for i in to_be_removed:
#     words.remove(i)
# wcloud = ''.join(words)
# cloud = jieba.lcut(wcloud)
# to_be_removed = []
# for i in cloud:
#     if i == '的':
#         to_be_removed.append(i)
#     elif i == '你':
#         to_be_removed.append(i)
#     elif i == '我':
#         to_be_removed.append(i)
#     elif i == '了':
#         to_be_removed.append(i)
#     elif i == '他':
#         to_be_removed.append(i)
#     elif i == '是':
#         to_be_removed.append(i)
#     elif i == '吗':
#         to_be_removed.append(i)
#     elif i == '去':
#         to_be_removed.append(i)
#     elif i == '有':
#         to_be_removed.append(i)
#     elif i == '吧':
#         to_be_removed.append(i)
# for i in to_be_removed:
#     cloud.remove(i)
# cloud = ' '.join(cloud)
# w = wordcloud.WordCloud(font_path='msyh.ttc', width=1000, height=700, background_color='white')
# w.generate(cloud)
# w.to_file('chen.png')

# import matplotlib.pyplot as plt
# import numpy as np
# x = np.arange(14,70,0.05)
# f = lambda x: (x-14)**5*np.exp((14-x)/2)/2**6/120
# plt.figure(figsize=(12,8), dpi=80)
# plt.plot(x,f(x),'r-')
# plt.grid(axis='y',alpha=0.6)
# plt.show()

