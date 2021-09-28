import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from .tools import train

class LitSimpleTrainer(pl.LightningModule):
    r"""[summary]

    Args:
        pl ([type]): [description]

    ---------
    Example 
    --------- ::

        arch = Vgg()
        dataset = FashionMNIST('./data', 128, 224)
        train_iter, test_iter = dataset()
        net = LitSimpleTrainer(arch,
                            torch.nn.CrossEntropyLoss(),
                            torch.optim.SGD,
                            1e-2
                            )
        train(net,train_iter,test_iter)
    """
    def __init__(self, arch, loss_fnc, optimizer_func, lr):
        super().__init__()
        self.model = arch
        self.loss_fnc = loss_fnc
        self.optimizer_func = optimizer_func
        self.lr = lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fnc(preds, y)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.parameters(), self.lr)
        return optimizer


