from torch import nn


class AlexNet(nn.Module):
    def __init__(self, input_channel=1, **kwargs):
        self.input_channel = input_channel
        super().__init__()
        self.encorder = nn.Sequential(
            nn.Conv2d(self.input_channel, 96, kernel_size=11,
                      stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.encorder(x)
        x = self.classifier(x)
        return x
