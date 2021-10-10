from torch import Tensor


def get_gen_loss(crit_fake_pred: Tensor) -> Tensor:
    return -crit_fake_pred.mean()


def get_crit_loss(crit_fake_pred: Tensor, crit_real_pred: Tensor,
                  gp: Tensor, c_lambda: float) -> Tensor:
    crit_loss = -(crit_real_pred.mean() -
                  crit_fake_pred.mean()) + c_lambda * gp.mean()
    return crit_loss
