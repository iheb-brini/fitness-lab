import torch
from Modules.nn.architectures._pytorch import DenseNet

def test():
    print('testing densenet...')
    X = torch.randn(size=(1, 3, 224, 224))
    net = DenseNet(in_channels=3)

    for layer in net.blocks:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    
    X = torch.randn(size=(1, 3, 224, 224))
    print(net(X).shape)

if __name__ == '__main__':
    test()
