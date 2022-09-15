import torch


class LinearSchedule:
    def __init__(self, T, beta_min=1e-4, beta_max=0.02):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, steps=T, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.zeros_like(self.alphas)

        a_hat = 1.0
        for i in range(0, T):
            a_hat = self.alphas[i] * a_hat
            self.alpha_hats[i] = a_hat

    def get_alphas(self, t):
        return self.alphas[torch.floor(t).long() - 1]

    def get_betas(self, t):
        return self.betas[torch.floor(t).long() - 1]

    def get_alpha_hats(self, t):
        t[t < 1] = 1
        t[t > self.T] = self.T
        return self.alpha_hats[torch.floor(t).long() - 1]

    def get_beta_hats(self, t):
        a_hat_t = self.get_alpha_hats(t)
        a_hat_tmin = self.get_alpha_hats(t - 1.0)
        return self.get_betas(t) * (1.0 - a_hat_tmin) / (1.0 - a_hat_t)
