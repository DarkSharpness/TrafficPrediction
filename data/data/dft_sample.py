import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# 原始数据
np.random.seed(114514)
x = np.arange(0, 365 * 24)
y = 3 * np.sin(2 * np.pi * x / (7 * 24)) + \
    4 * np.cos(2 * np.pi * x / (7 * 24)) + \
    -9 * np.sin(2 * np.pi * x / 24) + \
    10 * np.cos(2 * np.pi * x / 24)\

y = y * (1 + 0.5 * np.random.randn(len(x)))

# 已知的周期
periods = [7 * 24, 24]

# 插值方法
# 将 x 转换为一个周期内的值
x_mod = x % periods[0]

# 构建插值函数
interp_func = interp1d(x_mod, y, kind='linear', fill_value="extrapolate")

# 创建完整的 x 范围并转换为一个周期内的值
x_full = np.arange(0, max(x) + 1)
x_full_mod = x_full % periods[0]

# 使用插值函数计算完整的 y
y_full_interp = interp_func(x_full_mod)

# 周期性模型拟合方法
def periodic_model(x, *params):
    result = np.zeros_like(x, dtype=float)
    num_periods = len(params) // 4
    for i in range(num_periods):
        amplitude_sin = params[4 * i]
        phase_sin = params[4 * i + 1]
        amplitude_cos = params[4 * i + 2]
        phase_cos = params[4 * i + 3]
        period = periods[i]
        result += amplitude_sin * np.sin(2 * np.pi * x / period + phase_sin) + amplitude_cos * np.cos(2 * np.pi * x / period + phase_cos)
    return result

# 初始猜测参数
initial_params = [1.0, 0.0, 1.0, 0.0] * len(periods)

# 使用最小二乘法拟合参数
popt, _ = curve_fit(periodic_model, x, y, p0=initial_params)

# 使用拟合参数计算拟合值
y_full_fit = periodic_model(x_full, *popt)

# 绘图, 随机取 200 个点
plt.figure(figsize=(12, 6))
plt.plot(x_full[:200], y_full_interp[:200], '-', label='插值数据')
plt.plot(x_full[:200], y_full_fit[:200], '--', label='周期性拟合数据')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
