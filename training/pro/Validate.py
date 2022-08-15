import numpy as np
import torch
# import os
# from pro import Fn_Test
from pro.Loss import criterion_l2
# from torch.autograd import Variable

lsmx = torch.nn.LogSoftmax(dim=1)
fttype = torch.cuda.FloatTensor


def validate(model, val_loader, n_iter, val_loss, logwriter):
    model.eval()
    l_rmse = []
    # for sample in tqdm(val_loader):
    for sample in val_loader:
        M_mea = sample["spad"].type(fttype)
        dep = sample["targets"].type(fttype)
        dep_re, _ = model(M_mea)
        rmse = criterion_l2(dep_re, dep).data.cpu().numpy()
        l_rmse.append(rmse)
    # log the val losses
    logwriter.add_scalar("loss_val/rmse", np.mean(l_rmse), n_iter)
    val_loss["RMSE"].append(np.mean(l_rmse))
    print("Validation Loss--RMSE: {}".format(
        val_loss["RMSE"][len(val_loss["RMSE"]) - 1]))

    return val_loss, logwriter


def validate2(model, model_pre, val_loader, n_iter, val_loss, logwriter):
    model.eval()
    l_rmse = []
    # for sample in tqdm(val_loader):
    for sample in val_loader:
        M_mea = sample["spad"].type(fttype)
        dep = sample["targets"].type(fttype)
        _, rec128 = model_pre(M_mea)
        dep_re = model(rec128)
        rmse = criterion_l2(dep_re, dep).data.cpu().numpy()
        l_rmse.append(rmse)
    # log the val losses
    logwriter.add_scalar("loss_val/rmse", np.mean(l_rmse), n_iter)
    val_loss["RMSE"].append(np.mean(l_rmse))
    print("Validation Loss--RMSE: {}".format(
        val_loss["RMSE"][len(val_loss["RMSE"]) - 1]))

    return val_loss, logwriter
