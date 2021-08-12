from re import I
from mxnet import np,npx

from Modules import arch
from Modules.nn.architectures._mxnet import AlexNet

npx.set_np()

X = np.random.uniform(size=(1,1,224,224))

net = AlexNet()
net.initialize()

pred = net(X)
print(pred.shape)