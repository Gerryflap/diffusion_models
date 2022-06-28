import torch


class LinearSchedule:
    def __init__(self, T, beta_min=1e-4, beta_max=0.02):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, steps=T, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.zeros_like(self.alphas)

        a_hat = 1.0
        for i in range(0, T):
            a_hat = self.alphas[i]*a_hat
            self.alpha_hats[i] = a_hat

    def get_alphas(self, t):
        return self.alphas[torch.floor(t).long()-1]

    def get_betas(self, t):
        return self.betas[torch.floor(t).long()-1]

    def get_alpha_hats(self, t):
        return self.alpha_hats[torch.floor(t).long()-1]



