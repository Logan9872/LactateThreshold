import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# -------------------------------------数据读取和预处理----------------------------------------
# 遍历在文件夹中的文件并读取
dirs = os.listdir("C:\\Users\\Administrator\\Desktop\\blood_lactate\\")
for all_files in dirs:
    file = "C:\\Users\\Administrator\\Desktop\\blood_lactate\\" + all_files + ""
    CSV_data = pd.read_csv(
        file,
        skiprows=2,
        names=[
            'Velocity',
            'Lactate',
            'HuaweiHR',
            'PolarHR',
            'RPE'
        ],
        keep_default_na=False,
        encoding="unicode_escape")

    # 排除表格中的空数据
    while "" in CSV_data:  # 判断是否有空值在列表中
        CSV_data.remove("")  # 如果有就直接通过remove删除

    print(CSV_data)
    CSV_data = CSV_data.dropna(axis=0, how='all')

    x = CSV_data['Velocity']
    y = CSV_data['Lactate']
    h = CSV_data['HuaweiHR']
    hx = CSV_data['Velocity']

    # -------------------------------------数据计算和绘制----------------------------------------
    # 绘制双坐标轴的一侧
    plt.figure(figsize=(10, 6))
    # plt.xlim((3, 17))
    # plt.ylim((1, 7))

    ax1 = plt.gca()  # Get the current axis
    # 绘制散点图
    ax1.scatter(x, y, label='血乳酸样本')

    # 进行三阶多项式拟合
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)

    # 心率的三阶多项式拟合
    print(h)
    h1 = np.polyfit(hx, h, 3)
    hp = np.poly1d(h1)

    # 绘制拟合曲线

    x_fit = np.linspace(min(x), max(x), 100)
    hx_fit = np.linspace(min(hx), max(hx), 100)
    y_fit = p(x_fit)
    hy_fit = p(hx_fit)
    # 添加ax1中心率曲线的lable
    ax1.plot(x_fit, y_fit, c='lightcoral',  label='心率拟合曲线')

    ax1.plot(x_fit, y_fit, 'r', label='血乳酸拟合曲线')
    ax1.plot(hx_fit, hy_fit, 'r', alpha=0)

    # 绘制心率拟合曲线
    h_fit = hp(x_fit)
    ax2 = ax1.twinx()
    ax2.plot(hx_fit, h_fit, 'lightcoral', label='心率曲线')

    # 找到第一个数据点和最后一个数据点的坐标
    # DmaxMod找到两点间插值大于0.4的点连线
    for i in range(len(y)-1):
        point_distance = y[i+1] - y[i]
        if (point_distance >= 0.4):
            point_index = i
            print(point_distance, i)
            break

    first_point = (x[point_index], y[point_index])
    print(x, y, "111")
    last_point = (x[len(x)-1], y[len(y)-1])

    # 绘制直线连接第一个点和最后一个点
    ax1.plot([first_point[0], last_point[0]], [first_point[1], last_point[1]], 'g--', label="样本起始点连线")

    # 计算拟合曲线上每个点到直线的距离
    distances = []

    for i in range(len(x_fit)):
        # 计算点到直线的距离公式：|Ax + By + C| / sqrt(A^2 + B^2)
        # 这里遍历的是拟合的曲线到第一个和最后一个点连线的距离
        distance = ((last_point[0] - first_point[0]) * (first_point[1] - y_fit[i]) - (first_point[0] - x_fit[i]) * (
                    last_point[1] - first_point[1]))/np.sqrt((last_point[0] - first_point[0]) ** 2 + (last_point[1] - first_point[1]) ** 2)
        distances.append(distance)

    # 找到距离直线最远的点的索引
    # index_max_dist = np.argmax(distances)
    index_max_dist = distances.index(max(distances))

    # 获取距离直线最远的点的坐标
    x_max_dist = x_fit[index_max_dist]
    y_max_dist = y_fit[index_max_dist]

    # -------------------------------------绘制图例和美化图像-------------------------------------
    # 绘制HRn
    y_hr = hp(x_max_dist)

    # 在拟合曲线上标记Dmax值
    ax1.annotate(f'乳酸阈值:{y_max_dist:.2f} mmol/L', xy=(x_max_dist, y_max_dist), xytext=(x_max_dist - 2, y_max_dist + 1),
                 arrowprops=dict(arrowstyle='->', color='black'))

    # 乳酸阈心率标记
    # ax2.annotate(f'Huawei手表  乳酸阈心率:171 bpm, 乳酸阈跑速:9.42km/h\n直测法测量  乳酸阈心率:{y_hr:.0f} bpm, 乳酸阈跑速:{x_max_dist:.2f}km/h', xy=(x_max_dist, y_hr), xytext=(x_max_dist - 4, y_hr+7),
    #              arrowprops=dict(arrowstyle='->', color='black'))
    ax2.annotate(f'直测法测量  乳酸阈心率:{y_hr:.0f} bpm, 乳酸阈跑速:{x_max_dist:.2f}km/h', xy=(x_max_dist, y_hr), xytext=(x_max_dist - 4, y_hr+7),
                 arrowprops=dict(arrowstyle='->', color='black'))


    ax1.set_xlabel("跑速 (km/h)")
    ax1.set_ylabel("血乳酸值 (mmol/L)")
    ax2.set_ylabel("心率(bpm)")
    ax1.set_title("Dmax法测量乳酸阈值(三阶多项式函数拟合)", fontsize=16)
    ax1.legend()

    # -------------------------------------绘制的图片存储----------------------------------------
    title_png = all_files[:-4]  # 读取文件名".csv"之前的名称
    # 将filename 修改为路径
    filename = "C:/Users/Administrator/Desktop/乳酸实验/" + title_png + ".png"
    # 保存数据
    plt.savefig(filename)
    print(filename)
    plt.show()

