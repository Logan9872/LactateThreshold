import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 读取Excel文件
# df = pd.read_excel(r"C:\Users\Administrator\Desktop\hhh.xls", usecols=[0, 1])

data = {
    "intensity": [ 5.2, 6.4, 7.6, 8.8, 10, 11.2, 12.4, 13.6],
    # "lactate": [ 1.2, 1.6, 1.6, 2.4, 4.3, 2.6, 3.8, 6.0, 15.5]
    "lactate": [ 1.11, 1.16, 1.42, 1.83, 2.22, 2.93, 4.76, 6.68]

}

hr_data = {
"heart": [105, 106, 134, 150, 158, 168, 178, 184, 192],
    "intensity": [4, 5.2, 6.4, 7.6, 8.8, 10, 11.2, 12.4, 13.6]
}

df = pd.DataFrame(data)
HR_df = pd.DataFrame(hr_data )

x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
h = HR_df.iloc[:, 0].values
hx = HR_df.iloc[:, 1].values

# 绘制双坐标轴的一侧
plt.figure(figsize=(10, 6))
plt.xlim((3, 14))
plt.ylim((1, 7))

ax1 = plt.gca()  # Get the current axis
# 绘制散点图
ax1.scatter(x, y, label='血乳酸样本')

x_d = [4]
y_d = [1.64]
ax1.scatter(x_d, y_d, c="#1f77b4")

# 进行三阶多项式拟合
z = np.polyfit(x, y, 3)
p = np.poly1d(z)

# 心率的三阶多项式拟合
print(h)
h1 = np.polyfit(hx, h, 3)
hp = np.poly1d(h1)

# 进行指数函数拟合

# 绘制拟合曲线
# ax1.plot("",c= 'lightcoral',  label='心率拟合曲线')
x_fit = np.linspace(min(x), max(x), 100)
hx_fit = np.linspace(min(hx), max(hx), 100)
y_fit = p(x_fit)
hy_fit = p(hx_fit)
ax1.plot(x_fit, y_fit, 'r', label='血乳酸拟合曲线')
ax1.plot(hx_fit, hy_fit, 'r', alpha=0)



# 绘制心率拟合曲线
h_fit = hp(x_fit)

ax2 = ax1.twinx()
ax2.plot(hx_fit, h_fit, 'lightcoral', label='心率曲线')


print(h_fit)
# 找到第一个数据点和最后一个数据点的坐标 1
first_point = (x[0], y[0])
last_point = (x[-1], y[-1])

# 绘制直线连接第一个点和最后一个点
ax1.plot([first_point[0], last_point[0]], [first_point[1], last_point[1]], 'g--', label="样本起始点连线")

# 计算拟合曲线上每个点到直线的距离
distances = []

for i in range(len(x_fit)):
    # 计算点到直线的距离公式：|Ax + By + C| / sqrt(A^2 + B^2)

    # 这里我遍历的是拟合的曲线到第一个和最后一个点连线的距离 2
    distance = ((last_point[0] - first_point[0]) * (first_point[1] - y_fit[i]) - (first_point[0] - x_fit[i]) * (
                last_point[1] - first_point[1])) / np.sqrt((last_point[0] - first_point[0]) ** 2 + (last_point[1] - first_point[1]) ** 2)
    distances.append(distance)

# 在这里可以打印一下看看每个点到直线的距离 3
print("距离:", distances)

# 找到距离直线最远的点的索引
index_max_dist = np.argmax(distances)

# 获取距离直线最远的点的坐标
x_max_dist = x_fit[index_max_dist]
y_max_dist = y_fit[index_max_dist]


# 绘制HRn
y_hr = hp(x_max_dist)
print(y_hr)

# 在拟合曲线上标记Dmax值
ax1.annotate(f'乳酸阈值:{y_max_dist:.2f} mmol/L', xy=(x_max_dist, y_max_dist), xytext=(x_max_dist - 2, y_max_dist + 1),
             arrowprops=dict(arrowstyle='->', color='black'))

# 乳酸阈心率标记
ax2.annotate(f'Huawei手表  乳酸阈心率:171 bpm, 乳酸阈跑速:9.42km/h\n直测法测量  乳酸阈心率:{y_hr:.0f} bpm, 乳酸阈跑速:{x_max_dist:.2f}km/h', xy=(x_max_dist, y_hr), xytext=(x_max_dist - 4, y_hr+7),
             arrowprops=dict(arrowstyle='->', color='black'))


ax1.axhline(y=1.15, color='black', linestyle='--', label="基线(静息血乳酸)")
ax1.axhline(y=3.5, color='orange', linestyle='--', label="3.5 mmol/L乳酸阈")
ax1.axhline(y=4, color='purple', linestyle='--', label="4 mmol/L乳酸阈")
ax1.scatter(x=x_max_dist, y=y_max_dist, color='green', label="乳酸阈值点")
ax1.axvline(x=x_max_dist, color='black', linestyle='--',alpha=0.3)


ax1.set_xlabel("跑速 (km/h)")
ax1.set_ylabel("血乳酸值 (mmol/L)")
ax2.set_ylabel("心率(bpm)")
ax1.set_title("Dmax法测量乳酸阈值(三阶多项式函数拟合)", fontsize=16)
ax1.legend()
plt.show()

