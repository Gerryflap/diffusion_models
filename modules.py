import torch.nn
from torch.nn import Conv2d, GroupNorm, ConvTranspose2d
from torch.nn.functional import interpolate


class ResNetDownBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, downscale=False, use_norm=False, norm_groups=8, output=False):
        super().__init__()
        self.downscale = downscale
        self.output = output
        self.conv_1 = Conv2d(in_size, out_size, 3, padding=1, stride=2 if downscale else 1)
        self.conv_2 = Conv2d(out_size, out_size, 3, padding=1)

        self.ch_conv = None
        if in_size != out_size:
            self.ch_conv = Conv2d(in_size, out_size, 1)

        self.use_norm = use_norm
        self.norm = GroupNorm(norm_groups, out_size)
        self.norm2 = GroupNorm(norm_groups, out_size)

        # Assuming both the carried signal and the output of our layers are standard normal,
        #       we have to divide by sqrt(2) in order to make the sum of these 2 standard normal
        #       This constant is used to keep our signal in the correct scale
        self.const = 2.0 ** 0.5

    def forward(self, inp):
        x = inp

        x = self.conv_1(x)
        if self.use_norm:
            x = self.norm(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        if not self.output:
            if self.use_norm:
                x = self.norm2(x)
            x = torch.relu(x)

        if self.ch_conv is not None:
            inp = self.ch_conv(inp)

        if self.downscale:
            inp = interpolate(inp, scale_factor=0.5, mode="nearest")

        x = (inp + x) / self.const
        return x


class ResNetUpBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, upscale=False, use_norm=False, norm_groups=8, output=False):
        super().__init__()
        self.upscale = upscale
        self.output = output
        if upscale:
            self.conv_1 = ConvTranspose2d(in_size, out_size, 3, padding=1, output_padding=1, stride=2)
        else:
            self.conv_1 = ConvTranspose2d(in_size, out_size, 3, padding=1)
        self.conv_2 = ConvTranspose2d(out_size, out_size, 3, padding=1)

        self.ch_conv = None
        if in_size != out_size:
            self.ch_conv = Conv2d(in_size, out_size, 1)

        self.use_norm = use_norm
        self.norm = GroupNorm(norm_groups, out_size)
        self.norm2 = GroupNorm(norm_groups, out_size)

        # Assuming both the carried signal and the output of our layers are standard normal,
        #       we have to divide by sqrt(2) in order to make the sum of these 2 standard normal
        #       This constant is used to keep our signal in the correct scale
        self.const = 2.0 ** 0.5

    def forward(self, inp):
        x = inp

        x = self.conv_1(x)
        if self.use_norm:
            x = self.norm(x)
        x = torch.relu(x)

        x = self.conv_2(x)
        if not self.output:
            if self.use_norm:
                x = self.norm2(x)
            x = torch.relu(x)

        if self.ch_conv is not None:
            inp = self.ch_conv(inp)

        if self.upscale:
            inp = interpolate(inp, scale_factor=2.0, mode="nearest")

        x = (inp + x) / self.const
        return x

