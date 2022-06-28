import math

import torch


class CosineSchedule:
    def __init__(self, T):
        self.T = T

    def f(self, t):
        s = 0.008
        return torch.cos((math.pi / 2.0) * (t / self.T + s) / (1.0 + s)) ** 2.0

    def get_alpha_hats(self, t):
        return self.f(t) / self.f(torch.zeros_like(t))

    def get_betas(self, t):
        beta = 1.0 - self.get_alpha_hats(t) / self.get_alpha_hats(t - 1.0)
        # Beta is clipped in the paper in order to avoid issues near T
        return torch.clamp(beta, 0.0, 0.999)

    def get_beta_hats(self, t):
        a_hat_t = self.get_alpha_hats(t)
        a_hat_tmin = self.get_alpha_hats(t - 1.0)
        return self.get_betas(t) * (1.0 - a_hat_tmin) / (1.0 - a_hat_t)
