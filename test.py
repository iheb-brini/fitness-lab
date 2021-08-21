from mxnet import np,npx
from Modules.nn.architectures._mxnet import ResNet


npx.set_np()

X = np.random.uniform(size=(1, 3, 224, 224))

net = ResNet()
net.initialize()

for layer in net.blocks:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
