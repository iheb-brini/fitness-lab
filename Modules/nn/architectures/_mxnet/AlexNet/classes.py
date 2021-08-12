from mxnet.gluon import nn

class AlexNet(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encorder = nn.Sequential()
        self.encorder.add(
            nn.Conv2D(channels=96, kernel_size=(11, 11),
                      strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=(3, 3), strides=2),
            nn.Conv2D(channels=256, kernel_size=(5, 5),
                      padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=(3, 3), strides=2),
            nn.Conv2D(channels=384, kernel_size=(3, 3),
                      padding=1, activation='relu'),
            nn.Conv2D(channels=384, kernel_size=(3, 3),
                      padding=1, activation='relu'),
            nn.Conv2D(channels=256, kernel_size=(3, 3),
                      padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=(3, 3), strides=2),
        )

        self.classifier = nn.Sequential()
        self.classifier.add(
            nn.Dense(units=4096, activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(units=4096, activation='relu'),
            nn.Dropout(0.5),
            nn.Dense(units=1000),
        )

    def forward(self, x):
        return self.classifier(self.encorder(x))
