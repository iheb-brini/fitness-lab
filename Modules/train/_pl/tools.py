import pytorch_lightning as pl
from .constants import TRAINER_PARAMS

def train(net,train_iter,test_iter,params=TRAINER_PARAMS):
    trainer = pl.Trainer(**params)
    trainer.fit(net, train_iter, test_iter)