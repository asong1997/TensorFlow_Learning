""""
用一元线性回归解决回归问题： y = wx + b
"""
import numpy as np
import matplotlib.pyplot as plt
# 画图正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def compute_error(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # computer mean-squared-error
        totalError += (y - (w * x + b)) ** 2
    # average loss for each point
    return totalError / float(len(points))

def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 计算梯度 grad_b = 2(wx+b-y) grad_w = 2(wx+b-y)*x
        b_gradient += (2 / N) * ((w_current * x + b_current) - y)
        w_gradient += (2 / N) * x * ((w_current * x + b_current) - y)
    # update w'
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # update for several times
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]

def plot_scatter(data):
    x_data = data[:, 0]
    y_data = data[:, 1]
    plt.scatter(x_data, y_data)
    plt.title("训练数据集散点分布")
    plt.xlabel("自变量：x")
    plt.ylabel("因变量：y")
    plt.savefig("scatter.png")
    # plt.show()

def plot_result(data, w, b):
    x_data = data[:, 0]
    y_data = data[:, 1]
    plt.scatter(x_data, y_data, c='b')
    plt.plot(x_data, w * x_data + b, 'r')
    plt.title("训练拟合结果")
    plt.xlabel("自变量：x")
    plt.ylabel("因变量：y")
    plt.savefig("result.png")


def run():
    # numpy读取CSV文件
    points = np.genfromtxt("data.csv", delimiter=",")
    # 绘制数据散点图
    plot_scatter(points)
    # 设置学习率
    learning_rate = 0.0001
    # 权值初始化
    initial_b = 0
    initial_w = 0
    # 迭代次数
    num_iterations = 1000
    print("Starting b = {0}, w = {1}, error = {2}".format(initial_b, initial_w,
                                                          compute_error(initial_b, initial_w, points)))
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print(
        "After {0} iterations b = {1}, w = {2}, error = {3}".format(num_iterations, b, w, compute_error(b, w, points)))
    plot_result(points, w, b)


if __name__ == '__main__':
    run()
