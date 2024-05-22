import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def periodic_function(x, *params, periods):
    # 参数params中包含了所有正弦和余弦的系数，以及它们的频率
    y = np.zeros_like(x)
    n_periods = len(params) // 2
    for i in range(n_periods):
        amplitude = params[2*i]
        phase_shift = params[2*i+1]
        frequency = 2 * np.pi / periods[i]
        y += amplitude * np.sin(frequency * x + phase_shift) + amplitude * np.cos(frequency * x + phase_shift)
    return y

def sample(x, y, periods):
    # 为每个周期初始化振幅和相位
    initial_guess = []
    for _ in periods:
        initial_guess.append(1.0)  # 振幅的初始猜测
        initial_guess.append(0.0)  # 相位的初始猜测

    # 使用curve_fit来拟合周期函数
    # params, params_covariance = curve_fit(periodic_function, x, y, p0=initial_guess, args=(periods,))

    func = lambda x, *params: periodic_function(x, *params, periods=periods)

    params, _ = curve_fit(func, x, y, p0=initial_guess)

    # 返回拟合的参数
    return params

def plot_data_and_fit(x, y, periods, params):
    # 确保 x 是浮点数类型，以免类型转换错误
    x = np.asarray(x, dtype=float)

    x = x[100:200]
    y = y[100:200]

    # 生成拟合的y值
    y_fit = periodic_function(x, *params, periods=periods)

    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Original Data')  # 原始数据点
    plt.plot(x, y_fit, '-', label='Fitted Function')  # 拟合函数

    # 添加图例
    plt.legend()

    # 添加标题和坐标轴标签
    plt.title('Comparison of Original Data and Fitted Function')
    plt.xlabel('Time')
    plt.ylabel('Value')
    
    # 显示图表
    plt.show()

# 如果是 main
if __name__ == '__main__':
    # 原始数据
    np.random.seed(114514)
    x = np.arange(0, 365 * 24)
    y = 3 * np.sin(2 * np.pi * x / (7 * 24)) + \
        4 * np.cos(2 * np.pi * x / (7 * 24)) + \
        -9 * np.sin(2 * np.pi * x / 24) + \
        10 * np.cos(2 * np.pi * x / 24)\

    # 随机删除数据点
    mask = np.random.rand(len(x)) > 0.5
    x = x[mask]
    y = y[mask]

    y = y * (1 + 0.1 * np.random.randn(len(x)))

    # 已知的周期
    periods = [24, 2 * 24, 7 * 24, 365 * 24, 1]
    params = sample(x, y, periods)

    plot_data_and_fit(x, y, periods, params)
