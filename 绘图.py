from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# 设置黑色背景
plt.style.use('dark_background')

# 初始化参数
a, b, c = 1, 1, 1

# 创建初始数据
x = np.linspace(-10, 10, 500)
y = a * x**2 + b * x + c

# 创建图形和子图
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.3)  # 为滑块留出空间
line, = plt.plot(x, y, label="y = ax^2 + bx + c")
plt.xlim(-10, 10)
plt.ylim(-100, 1000)
plt.legend()

# 添加滑块
ax_a = plt.axes([0.1, 0.2, 0.8, 0.03])  # 滑块 a 的位置
ax_b = plt.axes([0.1, 0.15, 0.8, 0.03])  # 滑块 b 的位置
ax_c = plt.axes([0.1, 0.1, 0.8, 0.03])  # 滑块 c 的位置

slider_a = Slider(ax_a, 'a', -100.0, 100.0, valinit=a)
slider_b = Slider(ax_b, 'b', -100.0, 100.0, valinit=b)
slider_c = Slider(ax_c, 'c', -100.0, 100.0, valinit=c)

# 更新函数
def update(val):
    a = slider_a.val
    b = slider_b.val
    c = slider_c.val
    y = a * x**2 + b * x + c
    line.set_ydata(y)
    fig.canvas.draw_idle()

# 绑定滑块更新事件
slider_a.on_changed(update)
slider_b.on_changed(update)
slider_c.on_changed(update)

plt.show()