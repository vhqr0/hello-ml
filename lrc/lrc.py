import toolz as z
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# model

EPOCH = 10000
ETA = 1e-3


class M:
    def __init__(self, x, y):
        self.x_raw, self.y_raw = x, y
        self.x_mu, self.x_sigma, self.x = self.standardize(x)
        self.x = self.vectorlize(self.x)
        self.x_t = self.x.T
        self.y = np.array([self.y_raw]).T
        self.theta = np.random.rand(3, 1)

    @staticmethod
    def standardize(a):
        mu, sigma = a.mean(axis=0), a.std(axis=0)
        return mu, sigma, (a - mu) / sigma

    @staticmethod
    def vectorlize(a):
        return np.array([np.ones(len(a)), a[:, 0].T, a[:, 1].T]).T

    @property
    def f(self):
        return self.get_f(self.x)

    def get_f(self, x):
        return 1 / (1 + np.exp(-(x @ self.theta)))

    def __iter__(self):
        return self

    def __next__(self):
        self.theta -= ETA * (self.x_t @ (self.f - self.y))
        return self

    def plot(self):
        cord0 = self.y_raw == 0
        cord1 = self.y_raw == 1
        x = self.x[:, 1]
        cx = np.linspace(x.min(), x.max(), 100)
        cy = -(1 / self.theta[2]) * (self.theta[1] * cx + self.theta[0])
        plt.plot(self.x[cord0, 1], self.x[cord0, 2], "o")
        plt.plot(self.x[cord1, 1], self.x[cord1, 2], "x")
        plt.plot(cx, cy)


# train

matplotlib.use("tkagg")


def load_data(f):
    train = np.loadtxt(f, delimiter=",", skiprows=1)
    return train[:, 0:2], train[:, 2]


train_x, train_y = load_data("lrc/data.csv")

m = M(train_x, train_y)

plt.cla()
m.plot()
plt.show()

theta = z.thread_last(
    m,
    (z.take, EPOCH),
    (map, lambda x: x.theta.copy()),
    list,
    np.array,
)

plt.cla()
m.plot()
plt.show()
