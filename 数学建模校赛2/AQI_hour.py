import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.base.tsa_model
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'

# 加载数据
data1 = pd.read_excel('附件1：污染物浓度数据.xlsx')

data1 = data1.drop('质量等级', axis=1)
data1 = data1.fillna(data1.mean())
data1.index = pd.to_datetime(pd.date_range('1/1/2015','4/29/2023', freq='1d'))

# 数据集划分
train = data1.loc['6/1/2019':'4/29/2023']  # 2019年6月1日-2023年4月29日
test = data1.loc['1/1/2023':]  #2023年1月1日至今

label = 'AQI'
train = train[label]
test = test[label]

# 数据插条
time = pd.date_range('6/1/2019','4/29/2023', freq='1h')
tmp = pd.DataFrame(index=time, columns=time)
# 使用直线插值
for i in range(len(train) - 1):
    k = (train.iloc[i] - train.iloc[i+1]) / 24
    for j in range(24):
        tmp.iloc[j+24*i] = train.iloc[i] + k*(j + 1)
train = tmp
writer = train.ExcelWriter('插值.xlsx')
df.to_excel(writer, index=False)
writer.save()

# 计算ACF
fig2, ax2 = plt.subplots(figsize=(12,8), dpi=120)
plot_acf(train.diff(1).fillna(train.mean()), ax=ax2)
ax2.set_title('一阶差分后{}的自相关图'.format(label))
fig2.savefig('一阶差分后{}的自相关图'.format(label))
ax2.clear()
plot_acf(train, ax=ax2)
ax2.set_title('{}的自相关图'.format(label))
fig2.savefig(f'{label}的自相关图')

# 计算PACF
fig3, ax3 = plt.subplots(figsize=(12,8), dpi=120)
plot_pacf(train.diff(1).fillna(train.mean()), ax=ax3)
ax3.set_title(f"一阶差分后{label}的偏自相关图")
fig3.savefig(f'一阶差分后{label}的偏自相关图')
ax3.clear()
plot_pacf(train, ax=ax3)
ax3.set_title(f"{label}的偏自相关图")
fig3.savefig(f'{label}的偏自相关图')

# 模型建立
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train, order=(7, 1, 3), seasonal_order=(0,0,0,7))
# model = sm.tsa.arima.ARIMA(train, order=(6,1,2))
res = model.fit()
print(res.summary())

# 训练可视化
train_pred = res.predict(start="6/1/2019", end='4/29/2023')
fig7, ax7 = plt.subplots(figsize=(12,8), dpi=120)
ax7.plot(train.index, train_pred, 'r', alpha= 0.5, label='pred')
ax7.plot(train.index, train, 'b', alpha= 0.6, label='real')
fig7.legend()
fig7.savefig(f'{label}的训练可视化.jpg')

# 结果可视化
predict = res.predict(start='1/1/2023', end='4/29/2023')
fig4, ax4 = plt.subplots(figsize=(12, 8), dpi=120)
time_test = pd.date_range('1/1/2023', '4/29/2023')
ax4.set_title(f'{label}的7步模型预测')
ax4.plot(time_test, test, 'r', label='实际值', linewidth=3, alpha=0.5)
ax4.plot(time_test, predict, 'b', label='预测值', linewidth=3, alpha=0.6)
ax4.set_ylim([0, 200])
fig4.legend()
fig4.savefig(f'{label}的7步模型预测')

# RMES 计算
rmes = np.sqrt(np.mean((predict-test)**2))
print('RMES_pred: ', rmes)
rmes = np.sqrt(np.mean((train_pred-train)**2))
print('RtMES_train: ', rmes)

# 残差检验
residual = list(test - predict)
fig9, ax9 = plt.subplots(figsize=(8,8), dpi=120)
print('残差平均值: ', np.mean(residual))

ax9.hist(residual, bins=11)
fig9.savefig(f'{label}的残差检验')
ax9.clear()

from scipy import stats
stats.probplot(residual,plot=ax9)
fig9.savefig(f'{label}的残差正态性检验')

# 模型检验
from statsmodels.graphics.tsaplots import plot_predict
fig5, ax5 = plt.subplots(figsize=(12,8), dpi=120)
plot_predict(res, ax=ax5)
ax5.set_ylim([-200, 400])
fig5.savefig(f'{label}的置信区间诊断图')

fig8 = res.plot_diagnostics(figsize=(12,8))
fig8.savefig(f'{label}的模型检验')

# 未来12天预测
future = res.predict(start='4/30/2023',end='5/11/2023')
time_future = pd.date_range('4/30/2023','5/11/2023')
fig6, ax6 = plt.subplots(figsize=(12,8), dpi=120)
ax6.plot(time_future, future)
ax6.set_title(f'{label}的未来12天预测')
fig6.savefig(f'{label}的未来12天预测')
print('12天预测\n', future)
