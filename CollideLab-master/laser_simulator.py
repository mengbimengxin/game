import numpy as np
import numba
from matplotlib.pylab import plt
from timeit import timeit
from time import time

r = 10
x0, y0 = -10, 15

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)


def draw_circle(axes: plt.Axes, p_circle, R):
    delta = np.array([i / 180 * np.pi for i in range(360)])
    x = R * np.sin(delta) + p_circle[0]
    y = R * np.cos(delta) + p_circle[1]
    axes.plot(x, y, 'b')


def draw_laser(axes: plt.Axes):
    draw_laser_vectorize = np.vectorize(draw_one_laser_with_obs)
    delta = [j / 18 * np.pi for j in range(36)]
    draw_laser_vectorize(axes, delta)

    '''
    the version below without vectorized
    '''
    # for i in range(36):
    #     delta = i / 18 * np.pi
    #     draw_one_laser_with_obs(axes, delta)


# @numba.jit
def draw_one_laser_with_obs(axes: plt.Axes, delta):
    x = 30 * np.sin(delta)
    y = 30 * np.cos(delta)
    d = np.linalg.norm(np.array([x0, y0]))
    B1 = np.square(-2 * x0 - 2 * y0 * np.tan(delta))
    B2 = -4 * (-r ** 2 + x0 ** 2 + y0 ** 2) * (1 + np.tan(delta) ** 2)

    if np.linalg.norm(d / 30 * np.array([x, y]) - np.array([x0, y0])) <= r:
        A = 2 * x0 + 2 * y0 * np.tan(delta)
        B = np.sqrt(B1 + B2)
        C = 2 * (1 + np.tan(delta) ** 2)

        x = np.array([(A + B) / C, (A - B) / C])
        y = x * np.tan(delta)

        v1 = np.array([x[0], y[0]])
        v2 = np.array([x[1], y[1]])
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)

        v = v1 if d1 <= d2 else v2
        x = v[0]
        y = v[1]

        axes.plot([0, x], [0, y], '-r')
    else:
        axes.plot([0, x], [0, y], '-g')


def time_test():
    t = []
    draw_one_laser_with_obs_vec = np.vectorize(draw_one_laser_with_obs)
    delta = [j / 18 * np.pi for j in range(36)]
    for i in range(1500):
        t1 = time()
        draw_laser(ax)  # for迭代
        # draw_one_laser_with_obs_vec(ax, delta) #  向量化
        t2 = time()
        t.append(t2 - t1)

    del t[np.argmax(t)]
    print(" sum:{:.5f}\n avg:{:.15f}\n min:{:.5f}\n max:{:.5f}\n".format(sum(t), np.average(t), min(t), max(t)))
    plt.show()


if __name__ == '__main__':
    draw_circle(ax, [x0, y0], r)
    draw_circle(ax, [0, 0], np.linalg.norm(np.array([x0, y0])))
    draw_laser(ax)
    plt.show()
