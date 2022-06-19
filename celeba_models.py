import torch.nn

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

from torch import relu
from torch.nn import Conv2d, ConvTranspose2d


class UNetCelebA(torch.nn.Module):
    """
        Simple UNet-like convnet, with t input at the start.
        Skip connections are done via concatenation, on same spatial resolution levels.
    """

    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size

        self.conv_down_1 = Conv2d(3 + 1, h_size, 3, stride=2, padding=1)
        self.conv_down_2 = Conv2d(h_size, h_size * 2, 3, stride=2, padding=1)
        self.conv_down_3 = Conv2d(h_size * 2, h_size * 4, 3, stride=2, padding=1)
        self.conv_down_4 = Conv2d(h_size * 4, h_size * 8, 3, stride=2, padding=1)


        self.conv_middle_1 = Conv2d(h_size * 8, h_size * 8, 3, padding=1)
        self.conv_middle_2 = Conv2d(h_size * 8, h_size * 8, 3, padding=1)

        self.conv_up_1 = ConvTranspose2d(h_size * 16, h_size * 4, 3, stride=2, padding=1, output_padding=1)
        self.conv_up_2 = ConvTranspose2d(h_size * 8, h_size * 2, 3, stride=2, padding=1, output_padding=1)
        self.conv_up_3 = ConvTranspose2d(h_size * 4, h_size, 3, stride=2, padding=1, output_padding=1)
        self.conv_up_4 = ConvTranspose2d(h_size * 2, 3, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, t):
        t_vec = torch.ones_like(x)[:, :1] * t.view(-1, 1, 1, 1)
        x = torch.cat([x, t_vec], dim=1)

        x32 = self.conv_down_1(x)
        x32 = relu(x32)

        x16 = self.conv_down_2(x32)
        x16 = relu(x16)

        x8 = self.conv_down_3(x16)
        x8 = relu(x8)

        x4 = self.conv_down_4(x8)
        x4 = relu(x4)

        x = self.conv_middle_1(x4)
        x = relu(x)

        x = self.conv_middle_2(x)
        x = relu(x)

        x = self.conv_up_1(torch.cat([x, x4], dim=1))
        x = relu(x)

        x = self.conv_up_2(torch.cat([x, x8], dim=1))
        x = relu(x)

        x = self.conv_up_3(torch.cat([x, x16], dim=1))
        x = relu(x)

        x = self.conv_up_4(torch.cat([x, x32], dim=1))
        return x


if __name__ == "__main__":
    net = UNetCelebA(1)
    x = torch.normal(0, 1, (6, 3, 64, 64))
    t = torch.rand((6, 1))

    print(net(x, t).size())
