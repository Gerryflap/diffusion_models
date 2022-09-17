import math

import torch


class SinusoidalEmbedding(torch.nn.Module):
    """
        Creates sinusoidal embeddings of the given input.
        Assumes the input to be of size (-1, 1), where -1 denotes the batch dimension of arbitrary size.
        Will return embeddings of (-1, dim_out)
    """
    def __init__(self, dim_out=32, max_freq=1000.0, min_freq=1.0):
        super().__init__()
        if dim_out % 2 != 0:
            raise ValueError("dim_out should be divisible by 2, since there should be both sin and cos outputs")
        self.freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), dim_out // 2))
        factors = self.freqs * 2.0 * math.pi
        self.factors = torch.nn.Parameter(factors.view(1, dim_out // 2), requires_grad=False)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != 1:
            raise ValueError("Got invalid shape %s, input should be (-1, 1)" % str(x.size()))

        sines = torch.sin(self.factors * x)
        cosines = torch.cos(self.factors * x)
        return torch.cat([sines, cosines], dim=1)
