import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 输入数据
data = {
    "intensity": [0, 4, 5.2, 6.4, 7.6, 8.8, 10, 11.2, 12.4, 13.6],
    "lactate": [1.3, 1.2, 1.6, 1.6, 2.4, 4.3, 2.6, 3.8, 6.0, 15.5]
    # "lactate": [1.15, 1.64, 1.11, 1.16, 1.42, 1.83, 2.22, 2.93, 4.76, 6.68]
}

# 将数据转换为NumPy数组
intensity = np.array(data["intensity"])
lactate = np.array(data["lactate"])

# 定义三阶多项式函数
# def cubic_poly(x, a, b, c, d):
#     # return a * x**3 + b * x**2 + c * x + d
def cubic_poly(x, a, b, c):
    return a * x**2 + b * x + c


# 使用curve_fit拟合数据
params, covariance = curve_fit(cubic_poly, intensity, lactate)

# 解析参数
a, b, c= params

# 计算拟合后的乳酸曲线
fitted_lactate = cubic_poly(intensity, a, b, c)

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
Dmax_y = cubic_poly(Dmax_x, a, b, c)
print(f"Estimated Dmax: {Dmax_y:.2f} mmol/L")
print(f"Distance to line: {max_vertical_distance:.2f}")

# 绘制原始数据和拟合曲线
plt.scatter(intensity, lactate, label="Data")
plt.plot(intensity, fitted_lactate, 'r', label="Fit")

# 在拟合曲线上标记Dmax值
plt.annotate(f'Dmax: {Dmax_y:.2f} mmol/L', xy=(Dmax_x, Dmax_y), xytext=(Dmax_x - 2, Dmax_y + 2),
             arrowprops=dict(arrowstyle='->', color='black'))

# 绘制第一个数据点和最后一个数据点之间的连线
plt.plot([first_point[0], last_point[0]], [first_point[1], last_point[1]], 'g--', label="Line")

# 在Dmax点和连线上标记垂直距离
plt.annotate(f'Distance: {max_vertical_distance:.2f}', xy=(Dmax_x, Dmax_y), xytext=(Dmax_x - 4, Dmax_y - 2),
             arrowprops=dict(arrowstyle='->', color='blue'))

plt.xlabel("Intensity")
plt.ylabel("Lactate")
plt.legend()
plt.show()
