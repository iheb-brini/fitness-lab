from Modules.nn.architectures._pytorch.Vgg.classes import Vgg
import torch
from Modules.nn.architectures._pytorch import Vgg

X = torch.randn(size=(1, 1, 224, 224))

net = Vgg()

for layer in net.encoder:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

print(net.encoder[-1])
# pred = net(X)
# print(pred.shape)
