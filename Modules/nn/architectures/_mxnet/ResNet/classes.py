from mxnet.gluon import nn
from mxnet import np,npx

class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3,
                               padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None

        self.b1 = nn.BatchNorm()
        self.b2 = nn.BatchNorm()

    def forward(self, x):
        Y = npx.relu(self.b1(self.conv1(x)))
        Y = self.b2(self.conv2(Y))
        if self.conv3 is not None:
            x = self.conv3(x)

        return npx.relu(Y + x)


def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk


class ResNet(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.blocks = nn.Sequential()
        self.blocks.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3), nn.BatchNorm(),
                        nn.Activation('relu'), nn.MaxPool2D(pool_size=3, strides=2,
                                                            padding=1))
        self.blocks.add(resnet_block(64, 2, first_block=True), resnet_block(128, 2),
                        resnet_block(256, 2), resnet_block(512, 2))
        self.blocks.add(nn.GlobalAvgPool2D(), nn.Dense(10))

    def forward(self, x):
        return self.blocks(x)


