import torch.nn

# 28
# 14
# 7
# 4
# 4
# 7
# 14
# 28
from torch import relu
from torch.nn import Conv2d, ConvTranspose2d


class UNetMNIST(torch.nn.Module):
    """
        Simple UNet-like convnet, with t input at the start.
        Skip connections are done via concatenation, on same spatial resolution levels.
    """

    def __init__(self, h_size):
        super().__init__()
        self.h_size = h_size

        self.conv_down_1 = Conv2d(1 + 1, h_size, 3, stride=2, padding=1)
        self.conv_down_2 = Conv2d(h_size, h_size * 2, 3, stride=2, padding=1)
        self.conv_down_3 = Conv2d(h_size * 2, h_size * 4, 3, stride=2, padding=1)

        self.conv_middle_1 = Conv2d(h_size * 4, h_size * 4, 3, padding=1)
        self.conv_middle_2 = Conv2d(h_size * 4, h_size * 4, 3, padding=1)

        self.conv_up_1 = ConvTranspose2d(h_size * 8, h_size * 2, 3, stride=2, padding=1)
        self.conv_up_2 = ConvTranspose2d(h_size * 4, h_size, 3, stride=2, padding=1, output_padding=1)
        self.conv_up_3 = ConvTranspose2d(h_size * 2, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, t):
        t_vec = torch.ones_like(x) * t.view(-1, 1, 1, 1)
        x = torch.cat([x, t_vec], dim=1)

        x14 = self.conv_down_1(x)
        x14 = relu(x14)

        x7 = self.conv_down_2(x14)
        x7 = relu(x7)

        x4 = self.conv_down_3(x7)
        x4 = relu(x4)

        x = self.conv_middle_1(x4)
        x = relu(x)

        x = self.conv_middle_2(x)
        x = relu(x)

        x = self.conv_up_1(torch.cat([x, x4], dim=1))
        x = relu(x)

        x = self.conv_up_2(torch.cat([x, x7], dim=1))
        x = relu(x)

        x = self.conv_up_3(torch.cat([x, x14], dim=1))
        return x


if __name__ == "__main__":
    net = UNetMNIST(4)
    x = torch.normal(0, 1, (6, 1, 28, 28))
    t = torch.rand((6, 1))

    net(x, t)
