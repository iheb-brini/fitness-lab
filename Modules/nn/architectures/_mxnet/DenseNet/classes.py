from mxnet.gluon import nn
from mxnet import np

from .constants import NUM_CONVS_IN_DENSE_BLOCKS

def conv_block(num_channels):
    blk = nn.Sequential()

    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1)
            )

    return blk

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()

        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            # Concatenate the input and output of each block on the channel dimension
            X = np.concatenate((X, Y), axis=1)

        return X


def transitive_block(num_channels):
    blk = nn.Sequential()
    blk.add(
        nn.BatchNorm(), nn.Activation('relu'),
        nn.Conv2D(channels=num_channels,
                  kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2))
    return blk


class DenseNet(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = NUM_CONVS_IN_DENSE_BLOCKS

        self.blocks = nn.Sequential()
        self.blocks.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3), nn.BatchNorm(),
                        nn.Activation('relu'), nn.MaxPool2D(pool_size=3, strides=2,
                                                            padding=1))

        for i, num_conv in enumerate(num_convs_in_dense_blocks):
            self.blocks.add(DenseBlock(num_conv, growth_rate))

            num_channels += growth_rate*num_conv

            if i != len(num_convs_in_dense_blocks)-1:
                growth_rate //= 2
                self.blocks.add(transitive_block(num_channels))

        self.blocks.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(),
                        nn.Dense(10))

    def forward(self, X):
        return self.blocks(X)
