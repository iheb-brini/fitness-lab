from torch import nn
from .constants import CONV_ARCH
# %%


def vgg_block(num_conv, in_channels, out_channels):
    #blk = nn.Sequential()
    layers = []
    for _ in range(num_conv):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)


class Vgg(nn.Module):
    def __init__(self, in_channels=1,conv_arch=CONV_ARCH):
        super().__init__()
        layers = []
        for (num_conv, out_channels) in conv_arch:
            layers.append(
                vgg_block(num_conv, in_channels=in_channels,
                          out_channels=out_channels)
            )
            in_channels = out_channels

        layers.append(nn.Flatten())

        self.encoder = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=out_channels * 7 * 7, out_features=4096),
            nn.ReLU(), nn.Dropout2d(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(), nn.Dropout2d(p=0.5),
            nn.Linear(in_features=4096, out_features=10),
        )
        

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)
