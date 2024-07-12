import toolz as z
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# model

ETA = 1e-3
DELTA = 1e-2
MAXITER = 100


class M:
    def __init__(self, x, y):
        self.x_raw, self.y_raw = x, y
        self.x_mu, self.x_sigma, self.x = self.standardize(x)
        self.y_mu, self.y_sigma, self.y = self.standardize(y)
        self.theta0, self.theta1 = np.random.rand(), np.random.rand()
        self.e = self.get_e()

    @staticmethod
    def standardize(a):
        mu, sigma = a.mean(), a.std()
        return mu, sigma, (a - mu) / sigma

    @property
    def f(self):
        return self.get_f(self.x)

    def get_f(self, x):
        return self.theta0 + self.theta1 * x

    def get_e(self):
        return 0.5 * np.sum((self.y - self.f) ** 2)

    def get_f_raw(self, x_raw):
        x = (x_raw - self.x_mu) / self.x_sigma
        f = self.get_f(x)
        return f * self.y_sigma + self.y_mu

    def __iter__(self):
        return self

    def __next__(self):
        f_minus_y = self.f - self.y
        self.theta0 -= ETA * np.sum(f_minus_y)
        self.theta1 -= ETA * np.sum(f_minus_y * self.x)
        e = self.get_e()
        self.e, self.d = e, self.e - e
        return self

    def plot(self):
        fx = np.linspace(self.x.min(), self.x.max(), 100)
        fy = self.get_f(fx)
        plt.plot(self.x, self.y, "o")
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


train_x, train_y = load_data("data.csv")

m = M(train_x, train_y)

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
