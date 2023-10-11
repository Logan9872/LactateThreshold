import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# 输入数据
data = {
    "intensity": [ 4, 5.2, 6.4, 7.6, 8.8, 10, 11.2, 12.4, 13.6],
    # "lactate": [ 1.2, 1.6, 1.6, 2.4, 4.3, 2.6, 3.8, 6.0, 15.5]
    "lactate": [ 1.64, 1.11, 1.16, 1.42, 1.83, 2.22, 2.93, 4.76, 6.68]
}

# 将数据转换为NumPy数组
intensity = np.array(data["intensity"])
lactate = np.array(data["lactate"])

# 定义指数函数
def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

# 使用curve_fit拟合数据
params, covariance = curve_fit(exponential_func, intensity, lactate)

a, b, c = params

# 计算拟合后的乳酸曲线
fitted_lactate = exponential_func(intensity, a, b, c)

# 找到第一个数据点和最后一个数据点的坐标
first_point = (intensity[0], lactate[0])
last_point = (intensity[-1], lactate[-1])

# 初始化Dmax值和最大垂直距离
Dmax_x = None
max_vertical_distance = 0.0

# 计算拟合曲线上所有数据点到起始点连线的垂直距离
for x, y in zip(intensity, fitted_lactate):
    distance = np.abs((last_point[0]-first_point[0])*(first_point[1]-y) - (first_point[0]-x)*(last_point[1]-first_point[1])) / np.sqrt((last_point[0]-first_point[0])**2 + (last_point[1]-first_point[1])**2)
    if distance > max_vertical_distance:
        max_vertical_distance = distance
        Dmax_x = x

# 打印结果
Dmax_y = exponential_func(Dmax_x, a, b, c)
print(f"Estimated Dmax: {Dmax_y:.2f} mmol/L")
print(f"Distance to line: {max_vertical_distance:.2f}")

# 绘制原始数据和拟合曲线
plt.scatter(intensity, lactate, label="血乳酸样本")
plt.plot(intensity, fitted_lactate, 'r', label="血乳酸拟合曲线")

# 在拟合曲线上标记Dmax值
plt.annotate(f'Dmax: {Dmax_y:.2f} mmol/L', xy=(Dmax_x, Dmax_y), xytext=(Dmax_x - 1, Dmax_y + 1),
             arrowprops=dict(arrowstyle='->', color='black'))

# 绘制第一个数据点和最后一个数据点之间的连线
plt.plot([first_point[0], last_point[0]], [first_point[1], last_point[1]], 'g--', label="样本起始点连线")


plt.axhline(y=1.15, color='black', linestyle='--', label="基线(静息血乳酸)")
plt.axhline(y=3.5, color='orange', linestyle='--', label="3.5 mmol/L乳酸阈")
plt.axhline(y=4, color='purple', linestyle='--', label="4 mmol/L乳酸阈")
plt.scatter(x=Dmax_x, y=Dmax_y, color='green', label="乳酸阈值点")

plt.xlabel("跑速 (km/h)")
plt.ylabel("血乳酸值 (mmol/L)")
plt.title("DmaxMod法测量乳酸阈值(指数函数拟合)", fontsize=16)
plt.legend()
plt.show()
