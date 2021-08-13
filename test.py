import torch
from Modules.nn.architectures._pytorch import Vgg

X = torch.randn(size=(1, 3, 224, 224))

net = Vgg(in_channels=3)

pred = net(X)
print(pred.shape)
