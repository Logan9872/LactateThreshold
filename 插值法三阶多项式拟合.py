import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 读取Excel文件
# df = pd.read_excel(r"C:\Users\Administrator\Desktop\hhh.xls", usecols=[0, 1])

data = {
    "intensity": [4, 5.2, 6.4, 7.6, 8.8, 10, 11.2, 12.4, 13.6],
    # "lactate": [ 1.2, 1.6, 1.6, 2.4, 4.3, 2.6, 3.8, 6.0, 15.5]
    "lactate": [1.64, 1.11, 1.16, 1.42, 1.83, 2.22, 2.93, 4.76, 6.68],
    "heart": [105, 106, 134, 150, 158, 168, 178, 184, 192]
}

df = pd.DataFrame(data)

x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
z = df.iloc[:, 2].values

# 绘制散点图
plt.scatter(x, y, label='血乳酸样本')
# plt.scatter(x, z, label='心率样本')

# 进行三阶多项式拟合
z = np.polyfit(x, y, 3)
p = np.poly1d(z)

# 进行指数函数拟合


# 绘制拟合曲线
x_fit = np.linspace(min(x), max(x), 100)
y_fit = p(x_fit)
plt.plot(x_fit, y_fit, 'r', label='血乳酸拟合曲线')





# 找到第一个数据点和最后一个数据点的坐标 1
first_point = (x[0], y[0])
last_point = (x[-1], y[-1])

# 绘制直线连接第一个点和最后一个点
plt.plot([first_point[0], last_point[0]], [first_point[1], last_point[1]], 'g--', label="样本起始点连线")

# 计算拟合曲线上每个点到直线的距离
distances = []

for i in range(len(x_fit)):
    # 计算点到直线的距离公式：|Ax + By + C| / sqrt(A^2 + B^2)

    # 这里我遍历的是拟合的曲线到第一个和最后一个点连线的距离 2
    distance = np.abs((last_point[0] - first_point[0]) * (first_point[1] - y_fit[i]) - (first_point[0] - x_fit[i]) * (
                last_point[1] - first_point[1])) / np.sqrt((last_point[0] - first_point[0]) ** 2 + (last_point[1] - first_point[1]) ** 2)
    distances.append(distance)

# 在这里可以打印一下看看每个点到直线的距离 3
print("距离:", distances)

# 找到距离直线最远的点的索引
index_max_dist = np.argmax(distances)

# 获取距离直线最远的点的坐标
x_max_dist = x_fit[index_max_dist]
y_max_dist = y_fit[index_max_dist]



# 在拟合曲线上标记Dmax值
plt.annotate(f'Dmax: {y_max_dist:.2f} mmol/L,Speed:{x_max_dist:.2f}km/h', xy=(x_max_dist, y_max_dist ), xytext=(x_max_dist - 1, y_max_dist + 1),
             arrowprops=dict(arrowstyle='->', color='black'))


plt.axhline(y=1.15, color='black', linestyle='--', label="基线(静息血乳酸)")
plt.axhline(y=3.5, color='orange', linestyle='--', label="3.5 mmol/L乳酸阈")
plt.axhline(y=4, color='purple', linestyle='--', label="4 mmol/L乳酸阈")
plt.scatter(x=x_max_dist, y=y_max_dist, color='green', label="乳酸阈值点")

plt.xlabel("跑速 (km/h)")
plt.ylabel("血乳酸值 (mmol/L)")
plt.title("DmaxMod法测量乳酸阈值(指数函数拟合)", fontsize=16)
plt.legend()
plt.show()