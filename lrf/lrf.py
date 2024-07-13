import toolz as z
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# model

N = 5
ETA = 1e-3
DELTA = 1e-2
MAXITER = 100


class M:
    def __init__(self, n, x, y):
        self.n, self.x_raw, self.y_raw = n, x, y
        self.x_mu, self.x_sigma, self.x = self.standardize(x)
        self.y_mu, self.y_sigma, self.y = self.standardize(y)
        self.x = self.vectorlize(self.x)
        self.x_t = self.x.T
        self.y = np.array([self.y]).T
        self.theta = np.random.randn(n, 1)
        self.e = self.get_e()

    @staticmethod
    def standardize(a):
        mu, sigma = a.mean(), a.std()
        return mu, sigma, (a - mu) / sigma

    def vectorlize(self, a):
        return np.array([a**i for i in range(self.n)]).T

    @property
    def f(self):
        return self.get_f(self.x)

    def get_f(self, x):
        return x @ self.theta

    def get_e(self):
        return 0.5 * np.sum((self.y - self.f) ** 2)

    def get_f_raw(self, x_raw):
        x = (x_raw - self.x_mu) / self.x_sigma
        x = self.vectorlize(x)
        f = self.get_f(x)
        f = f[:, 0]
        return f * self.y_sigma + self.y_mu

    def __iter__(self):
        return self

    def __next__(self):
        self.theta -= ETA * (self.x_t @ (self.f - self.y))
        e = self.get_e()
        self.e, self.d = e, self.e - e
        return self

    def plot(self):
        x = self.x[:, 1]
        y = self.y[:, 0]
        fx = np.linspace(x.min(), x.max(), 100)
        fy = self.get_f(self.vectorlize(fx))[:, 0]
        plt.plot(x, y, "o")
        plt.plot(fx, fy)

    def plot_raw(self):
        fx_raw = np.linspace(self.x_raw.min(), self.x_raw.max(), 100)
        fy_raw = self.get_f_raw(fx_raw)
        plt.plot(self.x_raw, self.y_raw, "o")
        plt.plot(fx_raw, fy_raw)


# train

matplotlib.use("tkagg")


def load_data(f):
    train = np.loadtxt(f, delimiter=",", skiprows=1)
    return train[:, 0], train[:, 1]


train_x, train_y = load_data("lrf/data.csv")

m = M(N, train_x, train_y)

plt.cla()
m.plot()
plt.show()

e = z.thread_last(
    m,
    (itertools.takewhile, lambda x: x.d >= DELTA),
    (z.take, MAXITER),
    (map, lambda x: x.e),
    list,
    np.array,
)

plt.cla()
plt.plot(np.array(range(len(e))), e)
plt.show()

plt.cla()
m.plot_raw()
plt.show()
