import torch
from Modules.nn.architectures._pytorch import AlexNet

net = AlexNet(input_channel=3)

X= torch.randn(size=(1,3,224,224))



pred = net(X)   
print(pred.shape)