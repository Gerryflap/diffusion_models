import math
import torch


def f(t, T):
    s = 1e-6
    return torch.cos((math.pi / 2.0) * (t / T + s) / (1.0 + s)) ** 2.0


def alpha_hat(t, T):
    return f(t, T) / f(torch.zeros_like(t), T)


def beta(t, T):
    beta = 1.0 - alpha_hat(t, T)/alpha_hat(t-1, T)
    # Beta is clipped in the paper in order to avoid issues near T
    return beta.clamp(0, 0.999)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    xs = torch.linspace(0, 1000, 1000)
    ys = alpha_hat(xs, 1000)

    plt.plot(xs.numpy(), ys.numpy())
    plt.show()
