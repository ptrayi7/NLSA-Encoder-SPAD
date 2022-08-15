import torch


# import sys

def save_checkpoint(n_iter, epoch, model, optimer, file_path):
    """params:
    epcoh: the current epoch
    n_iter: the current iter
    model: the model dict
    optimer: the optimizer dict
    """
    state = {"n_iter": n_iter, "epoch": epoch, "lr": optimer.param_groups[0]["lr"], "state_dict": model.state_dict(),
             "optimizer": optimer.state_dict()}

    torch.save(state, file_path)


def save_checkpoint2(n_iter, epoch, model, model_pre, optimer, file_path):
    """params:
    epcoh: the current epoch
    n_iter: the current iter
    model: the model dict
    optimer: the optimizer dict
    """
    state = {"n_iter": n_iter, "epoch": epoch, "lr": optimer.param_groups[0]["lr"], "state_dict": model.state_dict(),
             "optimizer": optimer.state_dict(), "state_dict_depth": model_pre.state_dict()}

    torch.save(state, file_path)
