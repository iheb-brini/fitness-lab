import torch
from Modules.nn.architectures._pytorch import ResNet

def test():
    print('testing residual...')
    X = torch.randn(size=(1, 3, 224, 224))
    net = ResNet(in_channels=3)

    for layer in net.blocks:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    
    X = torch.randn(size=(1, 3, 224, 224))
    pred = net(X)
    print(pred.shape)


if __name__ == '__main__':
    test()
