from mxnet import np,npx
from Modules.nn.architectures._mxnet import DenseNet


npx.set_np()

X = np.random.uniform(size=(1, 1, 224, 224))

net = DenseNet()
net.initialize()

for layer in net.blocks:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
