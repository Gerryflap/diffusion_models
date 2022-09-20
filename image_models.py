"""
    Module containing the network architecture
"""
import math
import torch.nn
from torch import selu, relu
from torch.nn import Conv2d, ConvTranspose2d, Linear, GroupNorm
from modules import ResNetDownBlock, ResNetUpBlock
from sinusoidal_embedding import SinusoidalEmbedding


def transparent_norm(x):
    return x


def create_transparent_norm(s):
    return transparent_norm


@DeprecationWarning
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
            self.norm_lin_1 = GroupNorm(8, h_size * 16)
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


class UResNet(torch.nn.Module):
    """
        UNet-like Resnet, with t input at the start.
        By default this model will use sinusoidal embeddings of t
    """

    def __init__(self, h_size, resolution: int, use_norm=False, use_sin_embedding=True, embedding_size=32):
        super().__init__()
        self.h_size = h_size
        self.use_norm = use_norm

        if math.log2(resolution) != int(math.log2(resolution)) or resolution < 8:
            print("resolution should be a multiple of 2 and at the very least 8x8")

        # up/downscaling is done to 4x4, which is 2^2. So we need 'log2(resolution) - 2' up/downscaling layers
        self.downscales = int(math.log2(resolution)) - 2
        self.embedding_size = 32 if use_sin_embedding else 1
        self.sin_embedding = SinusoidalEmbedding(dim_out=embedding_size) if use_sin_embedding else None

        self.down_blocks = []
        for i in range(self.downscales):
            if i == 0:
                # Input block
                block = ResNetDownBlock(3 + self.embedding_size, h_size, downscale=True, use_norm=False)
            else:
                block = ResNetDownBlock(
                    int(h_size * (2 ** (i - 1))),
                    int(h_size * (2 ** i)),
                    downscale=True, use_norm=use_norm)
            self.down_blocks.append(block)

        self.linear_1 = Linear(
            int(4 * 4 * h_size * (2 ** (self.downscales - 1))),
            int(h_size * (2 ** self.downscales))
        )
        self.linear_2 = Linear(
            int(h_size * (2 ** self.downscales)),
            int(4 * 4 * h_size * (2 ** (self.downscales - 1)))
        )

        self.up_blocks = []
        for i in range(self.downscales - 1, -1, -1):
            block = ResNetUpBlock(
                int(2 * h_size * (2 ** i)),
                max(int(h_size * (2 ** (i - 1))), h_size),
                upscale=True, use_norm=use_norm)
            self.up_blocks.append(block)

        # Make the blocks visible to in parameters()
        self.down_blocks = torch.nn.ModuleList(self.down_blocks)
        self.up_blocks = torch.nn.ModuleList(self.up_blocks)

        # Not sure if 2 layers are needed, but otherwise there is a "path" through the model with
        #   no non-linearity from input to output (through first resnet block,
        #   then skipped to last resnet block, with no non-linearity in between
        # The conv_out1 layer will also process the full resolution input skip connection with t included
        self.conv_out1 = Conv2d(h_size + 3 + self.embedding_size, h_size, kernel_size=1)
        self.conv_out2 = Conv2d(h_size, 3, kernel_size=1)

        if use_norm:
            self.norm_lin_1 = GroupNorm(8, int(h_size * (2 ** self.downscales)))
            self.norm_lin_2 = GroupNorm(8, int(h_size * (2 ** (self.downscales - 1))))

    def forward(self, x, t):

        if self.sin_embedding is None:
            t_vec = torch.ones_like(x)[:, :1] * t.view(-1, 1, 1, 1)
        else:
            t_vec = torch.ones(x.size(0), self.embedding_size, x.size(2), x.size(3), device=x.device)
            t = self.sin_embedding(t)
            # Broadcast and fill vector
            t_vec[:, :, :, :] = t.view(-1, self.embedding_size, 1, 1)

        x = torch.cat([x, t_vec], dim=1)

        input_skip = x

        skip_connection_xs = []
        for i in range(self.downscales):
            x = self.down_blocks[i](x)
            skip_connection_xs.append(x)

        # Reverse list for upscaling order
        skip_connection_xs = skip_connection_xs[::-1]

        x = x.view(-1, 4 * 4 * self.h_size * (2 ** (self.downscales - 1)))
        x = self.linear_1(x)
        if hasattr(self, "norm_lin_1"):
            x = self.norm_lin_1(x)
        x = relu(x)

        x = self.linear_2(x)
        x = x.view(-1, self.h_size * (2 ** (self.downscales - 1)), 4, 4)
        if hasattr(self, "norm_lin_2"):
            x = self.norm_lin_2(x)
        x = relu(x)

        for i in range(self.downscales):
            x = torch.cat([x, skip_connection_xs[i]], dim=1)
            x = self.up_blocks[i](x)

        x = torch.cat([x, input_skip], dim=1)
        x = self.conv_out1(x)
        x = relu(x)

        x = self.conv_out2(x)
        return x


if __name__ == "__main__":
    # resolution check
    resolutions = [8, 16, 32, 64, 128]
    for res in resolutions:
        inp = torch.normal(0, 1, (10, 3, res, res))
        t = torch.rand((10, 1))
        model = UResNet(16, res, use_norm=True, use_sin_embedding=True)
        out = model(inp, t)
        print("Resolution: %d, output size: %s" % (res, out.size()))
        assert out.size(0) == 10 and out.size(1) == 3 and out.size(2) == res and out.size(3) == res
