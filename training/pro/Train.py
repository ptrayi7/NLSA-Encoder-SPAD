# The train function
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import scipy.io as scio
from pro import Validate, Fn_Test, Loss
from util import SaveChkp
import numpy as np

# import os

cudnn.benchmark = True
# 对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1。
lsmx = torch.nn.LogSoftmax(dim=1)
fttype = torch.cuda.FloatTensor
gradloss = Loss.GradLoss()


def train_nlsa_encoder(model_g, train_loader, val_loader, optimizer_g, epoch, n_iter,
                 val_loss, params, logwriter):
    flag_save = 0
    train_loss_epoch = {"loss_g": [], "loss_0": [], "loss_1": []}
    train_loss_ssim = {"ssim_d": [], "ssim_i": []}
    criterion_ssim = Loss.SSIM(data_range=1, channel=1)
    # with torch.no_grad():
    #     outdir_test = params["log_file"] + "/temp_" + str(epoch) + "_" + str(n_iter)
    #     Fn_Test.test_t(model_g, epoch, params, outdir_test)
    # for sample in train_loader:
    for sample in tqdm(train_loader):
        optimizer_g.zero_grad()
        n_iter += 1
        M_mea = sample["spad"].type(fttype)
        dep = sample["targets"].type(fttype)
        dep_re, _ = model_g(M_mea)
        rmse = Loss.criterion_l2(dep_re, dep)
        dep_re_d = torch.unsqueeze(dep_re[:, 0, :, :], 1)
        dep_re_i = torch.unsqueeze(dep_re[:, 1, :, :], 1)
        dep_d = torch.unsqueeze(dep[:, 0, :, :], 1)
        dep_i = torch.unsqueeze(dep[:, 1, :, :], 1)
        mae = Loss.criterion_l1(dep_re_d, dep_d)
        ssim_d = criterion_ssim(dep_re_d, dep_d)
        mae_i = Loss.criterion_l1(dep_re_i, dep_i)
        ssim_i = criterion_ssim(dep_re_i, dep_i)
        loss_g = (mae+0.2*(1-ssim_d)) + 0.025*(mae_i+0.25*(1-ssim_i))
        # loss_g = mae + 0.025 * mae_i
        train_loss_ssim["ssim_d"].append(ssim_d.data.cpu().numpy())
        train_loss_ssim["ssim_i"].append(ssim_i.data.cpu().numpy())
        loss_g.backward()
        optimizer_g.step()
        logwriter.add_scalar("loss_train/loss_g", loss_g, n_iter)
        train_loss_epoch["loss_g"].append(loss_g.data.cpu().numpy())
        logwriter.add_scalar("loss_train/loss_0", mae, n_iter)
        logwriter.add_scalar("loss_train/loss_1", rmse, n_iter)
        train_loss_epoch["loss_0"].append(mae.data.cpu().numpy())
        train_loss_epoch["loss_1"].append(rmse.data.cpu().numpy())
        if epoch == 1:
            loss_tv = Loss.criterion_tv(dep_re) / dep_re.size()[0]
            if loss_tv < 1:
                print("TV: " + str(loss_tv))
        if rmse > 0.3:
            print("RMSE: " + str(rmse))
        if n_iter % params["save_every"] == 0 and n_iter != 0:
            scio.savemat(file_name=params["log_file"] + "/train_loss_epoch" + str(epoch) + ".mat",
                         mdict=train_loss_epoch)
            with torch.no_grad():
                if epoch >= params["start_save_epoch"]:
                    outdir_test = params["log_file"] + "/temp_" + str(epoch) + "_" + str(n_iter)
                    flag_save = Fn_Test.test_t(model_g, epoch, params, outdir_test)
                    if flag_save == 1:
                        SaveChkp.save_checkpoint(n_iter, epoch, model_g, optimizer_g,
                                                 file_path=params["log_file"] + "/epoch_{}_{}.pth".format(epoch,
                                                                                                          n_iter))
        if n_iter >= 3 and params["all_test_flag"]:
            break

    # 跑完一次epoch计算一次val_loss
    with torch.no_grad():
        print("=========================>Start validation<=========================")
        val_loss, logwriter = Validate.validate(model_g, val_loader, n_iter, val_loss, logwriter)
        if epoch >= params["start_test_epoch"] or params["all_test_flag"]:
            outdir_test = params["log_file"] + "/temp_" + str(epoch) + "_" + str(n_iter)
            flag_save = Fn_Test.test_t(model_g, epoch, params, outdir_test)
        scio.savemat(file_name=params["log_file"] + "/train_loss.mat", mdict=train_loss_epoch)
        scio.savemat(file_name=params["log_file"] + "/val_loss.mat", mdict=val_loss)
        # save model states
        print("Validati complete\nSaving checkpoint...")
    if flag_save == 1:
        SaveChkp.save_checkpoint(n_iter, epoch, model_g, optimizer_g,
                                 file_path=params["log_file"] + "/epoch_{}_{}.pth".format(epoch, n_iter))
        print("Checkpoint saved!")
    print("SSIM_d: {}, SSIM_i:{}".format(1-np.mean(train_loss_ssim["ssim_d"]), 1-np.mean(train_loss_ssim["ssim_i"])))
    return model_g, optimizer_g, n_iter, train_loss_epoch, val_loss, logwriter
