import torch.nn

from torch import selu, relu
from torch.nn import Conv2d, ConvTranspose2d, Linear, GroupNorm

from modules import ResNetDownBlock, ResNetUpBlock
from sinusoidal_embedding import SinusoidalEmbedding


def transparent_norm(x):
    return x


def create_transparent_norm(s):
    return transparent_norm


class UResNet64(torch.nn.Module):
    """
        UNet-like Resnet, with t input at the start.
        By default this model will use sinusoidal embeddings of t
    """

    def __init__(self, h_size, use_norm=False, use_sin_embedding=True, embedding_size=32):
        super().__init__()
        self.h_size = h_size
        self.use_norm = use_norm

        self.embedding_size = 32 if use_sin_embedding else 1
        self.sin_embedding = SinusoidalEmbedding(dim_out=embedding_size) if use_sin_embedding else None

        self.conv_down_1 = ResNetDownBlock(3 + self.embedding_size, h_size, downscale=True, use_norm=False)
        self.conv_down_2 = ResNetDownBlock(h_size, h_size * 2, downscale=True, use_norm=use_norm)
        self.conv_down_3 = ResNetDownBlock(h_size * 2, h_size * 4, downscale=True, use_norm=use_norm)
        self.conv_down_4 = ResNetDownBlock(h_size * 4, h_size * 8, downscale=True, use_norm=use_norm)

        self.linear_1 = Linear(4 * 4 * h_size * 8, h_size * 16)
        self.linear_2 = Linear(h_size * 16, 4 * 4 * h_size * 8)

        self.conv_up_1 = ResNetUpBlock(h_size * 16, h_size * 4, upscale=True, use_norm=use_norm)
        self.conv_up_2 = ResNetUpBlock(h_size * 8, h_size * 2, upscale=True, use_norm=use_norm)
        self.conv_up_3 = ResNetUpBlock(h_size * 4, h_size, upscale=True, use_norm=use_norm)
        self.conv_up_4 = ResNetUpBlock(h_size * 2, h_size, upscale=True, use_norm=use_norm)

        # self.conv_out = ResNetDownBlock(h_size, 3, downscale=False, output=True, use_norm=False)
        self.conv_out = Conv2d(h_size, 3, kernel_size=1)

        if use_norm:
            self.norm_lin_1 = GroupNorm(8, h_size*16)
            self.norm_lin_2 = GroupNorm(8, h_size * 8)

    def forward(self, x, t):

        if self.sin_embedding is None:
            t_vec = torch.ones_like(x)[:, :1] * t.view(-1, 1, 1, 1)
        else:
            t_vec = torch.ones(x.size(0), self.embedding_size, x.size(2), x.size(3), device=x.device)
            t = self.sin_embedding(t)
            # Broadcast and fill vector
            t_vec[:, :, :, :] = t.view(-1, self.embedding_size, 1, 1)

        x = torch.cat([x, t_vec], dim=1)

        x32 = self.conv_down_1(x)

        x16 = self.conv_down_2(x32)

        x8 = self.conv_down_3(x16)

        x4 = self.conv_down_4(x8)

        x = x4.view(-1, 4 * 4 * self.h_size * 8)
        x = self.linear_1(x)
        if hasattr(self, "norm_lin_1"):
            x = self.norm_lin_1(x)
        x = relu(x)

        x = self.linear_2(x)
        x = x.view(-1, self.h_size * 8, 4, 4)
        if hasattr(self, "norm_lin_2"):
            x = self.norm_lin_2(x)
        x = relu(x)

        x = self.conv_up_1(torch.cat([x, x4], dim=1))

        x = self.conv_up_2(torch.cat([x, x8], dim=1))

        x = self.conv_up_3(torch.cat([x, x16], dim=1))

        x = self.conv_up_4(torch.cat([x, x32], dim=1))

        x = self.conv_out(x)
        return x

