from torch import nn, cat

from .constants import NUM_CONVS_IN_DENSE_BLOCKS


def conv_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, padding=1)
    )

    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        block_list = []
        for i in range(num_convs):
            block_list.append(conv_block(
                out_channels*i + in_channels, out_channels))

        self.net = nn.Sequential(*block_list)

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            # Concatenate the input and output of each block on the channel dimension
            X = cat((X, Y), axis=1)

        return X


def transitive_block(in_channels, out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels), nn.ReLU(),
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return blk


class DenseNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__(**kwargs)

        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = NUM_CONVS_IN_DENSE_BLOCKS

        list_blocks = [
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            list_blocks.append(DenseBlock(
                num_convs, num_channels, growth_rate))

            num_channels += num_convs * growth_rate

            if i != len(num_convs_in_dense_blocks) - 1:
                list_blocks.append(transitive_block(
                    num_channels, num_channels // 2))
                num_channels = num_channels // 2

        list_blocks.extend([nn.BatchNorm2d(num_channels), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(), nn.Linear(num_channels, 10)])

        self.blocks = nn.Sequential(*list_blocks)

    def forward(self, X):
        return self.blocks(X)
