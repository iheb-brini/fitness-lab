from re import I
from mxnet import np,npx

from Modules.nn.architectures._mxnet import AlexNet
from Modules.nn.architectures._mxnet import Vgg

npx.set_np()

X = np.random.uniform(size=(1,1,224,224))

net = Vgg()
net.initialize()

pred = net(X)
print(pred.shape)