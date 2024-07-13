import toolz as z
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# model

EPOCH = 10


class M:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.w = np.zeros(2)

    def get_f(self, x):
        if self.w @ x >= 0:
            return 1
        else:
            return -1

    def __iter__(self):
        return self

    def __next__(self):
        for x, y in zip(self.x, self.y):
            if self.get_f(x) != y:
                self.w += y * x
        return self

    def plot(self):
        cord0 = self.y == -1
        cord1 = self.y == 1
        x = self.x[:, 0]
        cx = np.linspace(x.min(), x.max(), 100)
        cy = -(self.w[0] / self.w[1]) * cx
        plt.plot(self.x[cord0, 0], self.x[cord0, 1], "o")
        plt.plot(self.x[cord1, 0], self.x[cord1, 1], "x")
        plt.plot(cx, cy)


# train

matplotlib.use("tkagg")


def load_data(f):
    train = np.loadtxt(f, delimiter=",", skiprows=1)
    return train[:, 0:2], train[:, 2]


train_x, train_y = load_data("lrc/data.csv")
train_y[train_y == 0] = -1  # perc use 1 and -1

m = M(train_x, train_y)

plt.cla()
m.plot()
plt.show()

w = z.thread_last(
    m,
    (z.take, EPOCH),
    (map, lambda x: x.w.copy()),
    list,
    np.array,
)

plt.cla()
m.plot()
plt.show()
