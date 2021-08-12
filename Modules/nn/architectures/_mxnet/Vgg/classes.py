from mxnet.gluon import nn
from .constants import CONV_ARCH
#%%
def vgg_block(num_conv, num_channels):
    blk = nn.Sequential()
    for _ in range(num_conv):
        blk.add(
            nn.Conv2D(num_channels, kernel_size=3,
                      padding=1, activation='relu')
        )

    blk.add(nn.MaxPool2D(pool_size=2, strides=2))

    return blk


class Vgg(nn.Block):
    def __init__(self, conv_arch=CONV_ARCH, **kwargs):
        super().__init__(**kwargs)
        self.encoder = nn.Sequential()
        for (num_conv, num_channels) in conv_arch:
            self.encoder.add(
                vgg_block(num_conv, num_channels)
            )
        self.classifier = nn.Sequential()

        self.classifier.add(
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(1000)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)