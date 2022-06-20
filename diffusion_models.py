import math
import torch


def f(t, T):
    s = 0.008
    return torch.cos((math.pi / 2.0) * (t / T + s) / (1.0 + s)) ** 2.0


def alpha_hat(t, T):
    return f(t, T) / f(torch.zeros_like(t), T)


def beta(t, T):
    beta = 1.0 - alpha_hat(t, T) / alpha_hat(t - 1.0, T)
    # Beta is clipped in the paper in order to avoid issues near T
    return torch.clamp(beta, 0.0, 0.999)


def beta_hat(t, T):
    a_hat_t = alpha_hat(t, T)
    a_hat_tmin = alpha_hat(t - 1.0, T)
    return beta(t, T) * (1.0 - a_hat_tmin) / (1.0 - a_hat_t)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    xs = torch.linspace(0, 1000, 1000)
    ys = alpha_hat(xs, 1000)
    bs = beta(xs, 1000)

    plt.plot(xs.numpy(), ys.numpy())
    plt.plot(xs.numpy(), bs.numpy())
    plt.show()
