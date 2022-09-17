import torch.nn
"""
    Legacy model definitions for loading the old trained models
"""

# 64
# 32
# 16
# 8
# 4
# 4
# 8
# 16
# 32
# 64

from torch import selu, relu
from torch.nn import Conv2d, ConvTranspose2d, Linear, GroupNorm

from modules import ResNetDownBlock, ResNetUpBlock


def transparent_norm(x):
    return x


def create_transparent_norm(s):
    return transparent_norm


class UNetCelebA(torch.nn.Module):
    """
        Simple UNet-like convnet, with t input at the start.
        Skip connections are done via concatenation, on same spatial resolution levels.
    """

    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size

        self.conv_down_1 = Conv2d(3 + 1, h_size, 3, stride=2, padding=1, bias=True)
        self.conv_down_2 = Conv2d(h_size, h_size * 2, 3, stride=2, padding=1, bias=True)
        self.conv_down_3 = Conv2d(h_size * 2, h_size * 4, 3, stride=2, padding=1, bias=True)
        self.conv_down_4 = Conv2d(h_size * 4, h_size * 8, 3, stride=2, padding=1, bias=True)

        self.conv_middle_1 = Conv2d(h_size * 8, h_size * 16, 4, bias=True)
        self.conv_middle_2 = ConvTranspose2d(h_size * 16 + 1, h_size * 8, 4, bias=True)

        self.conv_up_1 = ConvTranspose2d(h_size * 16, h_size * 4, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_up_2 = ConvTranspose2d(h_size * 8, h_size * 2, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_up_3 = ConvTranspose2d(h_size * 4, h_size, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_up_4 = ConvTranspose2d(h_size * 2, h_size, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.conv_out = Conv2d(h_size, 3, 3, padding=1, bias=False)

        self.output_bias = torch.nn.Parameter(torch.zeros((3, 64, 64)), requires_grad=True)

        # norm_fn = lambda s: GroupNorm(8, s)
        norm_fn = create_transparent_norm
        self.norm_1 = norm_fn(h_size)
        self.norm_2 = norm_fn(h_size * 2)
        self.norm_3 = norm_fn(h_size * 4)
        self.norm_4 = norm_fn(h_size * 8)

        self.norm_5 = norm_fn(h_size * 8)
        self.norm_6 = norm_fn(h_size * 8)

        self.norm_7 = norm_fn(h_size * 4)
        self.norm_8 = norm_fn(h_size * 2)
        self.norm_9 = norm_fn(h_size)
        self.norm_10 = norm_fn(h_size)

    def forward(self, x, t):
        t_vec = torch.ones_like(x)[:, :1] * t.view(-1, 1, 1, 1)
        x = torch.cat([x, t_vec], dim=1)

        x32 = self.conv_down_1(x)
        x32 = self.norm_1(x32)
        x32 = selu(x32)

        x16 = self.conv_down_2(x32)
        x16 = self.norm_2(x16)
        x16 = selu(x16)

        x8 = self.conv_down_3(x16)
        x8 = self.norm_3(x8)
        x8 = selu(x8)

        x4 = self.conv_down_4(x8)
        x4 = self.norm_4(x4)
        x4 = selu(x4)

        x = self.conv_middle_1(x4)
        x = self.norm_5(x)
        x = selu(x)

        x = self.conv_middle_2(torch.cat([x, t.view(-1, 1, 1, 1)], dim=1))
        x = self.norm_6(x)
        x = selu(x)

        x = self.conv_up_1(torch.cat([x, x4], dim=1))
        x = self.norm_7(x)
        x = selu(x)

        x = self.conv_up_2(torch.cat([x, x8], dim=1))
        x = self.norm_8(x)
        x = selu(x)

        x = self.conv_up_3(torch.cat([x, x16], dim=1))
        x = self.norm_9(x)
        x = selu(x)

        x = self.conv_up_4(torch.cat([x, x32], dim=1))
        x = self.norm_10(x)
        x = selu(x)

        x = self.conv_out(x)
        x += self.output_bias
        return x


class UResNetCelebA(torch.nn.Module):
    """
        UNet-like Resnet, with t input at the start. Same as the model above, but for 32x32
        Skip connections are done via concatenation, on same spatial resolution levels.
    """

    def __init__(self, h_size, use_norm=False):
        super().__init__()
        self.h_size = h_size
        self.use_norm = use_norm

        self.conv_down_1 = ResNetDownBlock(4, h_size, downscale=True, use_norm=False)
        self.conv_down_2 = ResNetDownBlock(h_size, h_size * 2, downscale=True, use_norm=use_norm)
        self.conv_down_3 = ResNetDownBlock(h_size * 2, h_size * 4, downscale=True, use_norm=use_norm)
        self.conv_down_4 = ResNetDownBlock(h_size * 4, h_size * 8, downscale=True, use_norm=use_norm)

        self.linear_1 = Linear(4 * 4 * h_size * 8, h_size * 16)
        self.linear_2 = Linear(h_size * 16 + 1, 4 * 4 * h_size * 8)

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
        t_vec = torch.ones_like(x)[:, :1] * t.view(-1, 1, 1, 1)
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

        x = self.linear_2(torch.cat([x, t], dim=1))
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


class UNetCelebA32(torch.nn.Module):
    """
        Simple UNet-like convnet, with t input at the start. Same as the model above, but for 32x32
        Skip connections are done via concatenation, on same spatial resolution levels.
    """

    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size

        self.conv_down_1 = Conv2d(3 + 1, h_size, 3, stride=2, padding=1, bias=True)
        self.conv_down_2 = Conv2d(h_size, h_size * 2, 3, stride=2, padding=1, bias=True)
        self.conv_down_3 = Conv2d(h_size * 2, h_size * 4, 3, stride=2, padding=1, bias=True)

        self.linear_1 = Linear(4 * 4 * h_size * 4, h_size * 8)
        self.linear_2 = Linear(h_size * 8 + 1, 4 * 4 * h_size * 4)

        self.conv_up_2 = ConvTranspose2d(h_size * 8, h_size * 2, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_up_3 = ConvTranspose2d(h_size * 4, h_size, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv_up_4 = ConvTranspose2d(h_size * 2, h_size, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.conv_out = Conv2d(h_size, 3, 3, padding=1, bias=False)

        # norm_fn = lambda s: GroupNorm(8, s)
        # norm_fn = lambda s: BatchNorm2d(s)

        # Transparent function for no normalization
        norm_fn = lambda s: (lambda h: h)

        self.norm_1 = norm_fn(h_size)
        self.norm_2 = norm_fn(h_size * 2)
        self.norm_3 = norm_fn(h_size * 4)

        self.norm_4 = norm_fn(h_size * 4)
        self.norm_5 = norm_fn(h_size * 4)

        self.norm_6 = norm_fn(h_size * 2)
        self.norm_7 = norm_fn(h_size)
        self.norm_8 = norm_fn(h_size)

        self.output_bias = torch.nn.Parameter(torch.zeros((3, 32, 32)), requires_grad=True)

    def forward(self, x, t):
        t_vec = torch.ones_like(x)[:, :1] * t.view(-1, 1, 1, 1)
        x = torch.cat([x, t_vec], dim=1)

        x16 = self.conv_down_1(x)
        x16 = self.norm_1(x16)
        x16 = selu(x16)

        x8 = self.conv_down_2(x16)
        x8 = self.norm_2(x8)
        x8 = selu(x8)

        x4 = self.conv_down_3(x8)
        x4 = self.norm_3(x4)
        x4 = selu(x4)

        x = x4.view(-1, 4 * 4 * self.h_size * 4)
        x = self.linear_1(x)
        x = self.norm_4(x)
        x = selu(x)

        x = self.linear_2(torch.cat([x, t], dim=1))
        x = self.norm_5(x)
        x = selu(x)
        x = x.view(-1, self.h_size * 4, 4, 4)

        x = self.conv_up_2(torch.cat([x, x4], dim=1))
        x = self.norm_6(x)
        x = selu(x)

        x = self.conv_up_3(torch.cat([x, x8], dim=1))
        x = self.norm_7(x)
        x = selu(x)

        x = self.conv_up_4(torch.cat([x, x16], dim=1))
        x = self.norm_8(x)
        x = selu(x)

        x = self.conv_out(x)
        x = self.output_bias + x
        return x


class UResNetCelebA32(torch.nn.Module):
    """
        Simple UNet-like Resnet, with t input at the start. Same as the model above, but for 32x32
        Skip connections are done via concatenation, on same spatial resolution levels.
    """

    def __init__(self, h_size, use_norm=False):
        super().__init__()
        self.h_size = h_size

        self.conv_down_1 = ResNetDownBlock(4, h_size, downscale=True, use_norm=use_norm)
        self.conv_down_2 = ResNetDownBlock(h_size, h_size * 2, downscale=True, use_norm=use_norm)
        self.conv_down_3 = ResNetDownBlock(h_size * 2, h_size * 4, downscale=True, use_norm=use_norm)

        self.linear_1 = Linear(4 * 4 * h_size * 4, h_size * 8)
        self.linear_2 = Linear(h_size * 8 + 1, 4 * 4 * h_size * 4)

        self.conv_up_2 = ResNetUpBlock(h_size * 8, h_size * 2, upscale=True, use_norm=use_norm)
        self.conv_up_3 = ResNetUpBlock(h_size * 4, h_size, upscale=True, use_norm=use_norm)
        self.conv_up_4 = ResNetUpBlock(h_size * 2, h_size, upscale=True, use_norm=use_norm)

        self.conv_out = ResNetDownBlock(h_size, 3, downscale=False, output=True, use_norm=False)

        self.output_bias = torch.nn.Parameter(torch.zeros((3, 32, 32)), requires_grad=True)

    def forward(self, x, t):
        t_vec = torch.ones_like(x)[:, :1] * t.view(-1, 1, 1, 1)
        x = torch.cat([x, t_vec], dim=1)

        x16 = self.conv_down_1(x)

        x8 = self.conv_down_2(x16)

        x4 = self.conv_down_3(x8)

        x = x4.view(-1, 4 * 4 * self.h_size * 4)
        x = self.linear_1(x)
        x = relu(x)

        x = self.linear_2(torch.cat([x, t], dim=1))
        x = relu(x)
        x = x.view(-1, self.h_size * 4, 4, 4)

        x = self.conv_up_2(torch.cat([x, x4], dim=1))

        x = self.conv_up_3(torch.cat([x, x8], dim=1))

        x = self.conv_up_4(torch.cat([x, x16], dim=1))

        x = self.conv_out(x)
        x = self.output_bias + x
        return x


if __name__ == "__main__":
    net = UResNetCelebA(8)
    x = torch.normal(0, 1, (6, 3, 64, 64))
    t = torch.rand((6, 1))

    print(net(x, t).size())

    net = UResNetCelebA32(8)
    x = torch.normal(0, 1, (6, 3, 32, 32))
    t = torch.rand((6, 1))

    print(net(x, t).size())
