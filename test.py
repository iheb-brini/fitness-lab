from Modules.nn.architectures._pytorch import Vgg
from Modules.data.dataset import FashionMNIST
from Modules.train._pl import (LitSimpleTrainer, train)
import torch


def test_train_pl():
    arch = Vgg()
    dataset = FashionMNIST('./data', 16, 224)
    train_iter, test_iter = dataset()
    net = LitSimpleTrainer(arch,
                           torch.nn.CrossEntropyLoss(),
                           torch.optim.SGD,
                           1e-2
                           )
    train(net,train_iter,test_iter)


if __name__ == '__main__':
    test_train_pl()
