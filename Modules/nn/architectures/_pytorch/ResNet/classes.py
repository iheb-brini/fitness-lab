from torch import nn
import torch

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=stride)
        else:
            self.conv3 = None

        self.b1 = nn.BatchNorm2d(out_channels)
        self.b2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        Y = self.conv1(x)
        Y = torch.relu(self.b1(Y))
        Y = self.b2(self.conv2(Y))
        if self.conv3 is not None:
            x = self.conv3(x)

        return torch.relu(Y + x)



def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    block_list = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block_list.append(
                Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            block_list.append(Residual(out_channels, out_channels))

    return nn.Sequential(*block_list)


class ResNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__(**kwargs)
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet_block(64, 64, 2, first_block=True),
            resnet_block(64, 128, 2),
            resnet_block(128, 256, 2), resnet_block(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        return self.blocks(x)