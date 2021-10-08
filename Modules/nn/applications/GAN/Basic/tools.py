from torch import (optim, Tensor, zeros_like, ones_like,)
from .classes import Generator, Discriminator


def get_disc_loss(gen: Generator, disc: Discriminator,
                  criterion: optim,
                  real: Tensor,
                  noise: Tensor):

    fake = gen(noise)

    pred_fake = disc(fake.detach())
    loss_fake = criterion(pred_fake, zeros_like(pred_fake))

    pred_real = disc(real)
    loss_real = criterion(pred_real, ones_like(pred_real))

    return (loss_real + loss_fake) / 2


def get_gen_loss(gen: Generator, disc: Discriminator,
                 criterion: optim,
                 noise: Tensor):
    fake = gen(noise)
    pred = disc(fake)
    loss = criterion(pred, ones_like(pred))
    return loss
