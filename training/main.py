# The train file for network
import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from util.SpadDataset import SpadDataset
from util.ParseArgs import parse_args
from pro import Train
from models import model_b4_DDFN_CGNL, model_DRNLSA, model_DRNLSA_AE, model_unet, model_NLS_Encoder
import scipy.io as scio
import os
import warnings
import time

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main():
    opt = parse_args("./config.ini")

    if opt["use_gpu"]:
        print("Number of available GPUs: {} {}".format(torch.cuda.device_count(),
                                                       torch.cuda.get_device_name(torch.cuda.current_device())))
        torch.cuda.empty_cache()
    else:
        print("USE CPU")
    print("Batch_size: {}\n".format(opt["batch_size"]),
          "Workers: {}\n".format(opt["workers"]),
          "All_test_flag: {}\n".format(opt["all_test_flag"]),
          "Train_target: {}\n".format(opt["train_target"]),
          "Train_size: {}\n".format(opt["train_size"]),
          "Val_size: {}\n".format(opt["val_size"]),
          "Epoch: {}\n".format(opt["epoch"]),
          "Save_every: {}\n".format(opt["save_every"]),
          "Noise_idx: {}\n".format(opt["noise_idx"]),
          "Training_path: {}\n".format(opt["training_path"]),
          "Log_dir: {}\n".format(opt["log_dir"]),
          "Log_file: {}\n".format(opt["log_file"]),
          "Util_dir: {}\n".format(opt["util_dir"]),
          "Train_file: {}\n".format(opt["train_file"]),
          "Val_file: {}\n".format(opt["val_file"]),
          "test_data_file_path:{}".format(opt["test_dist_file"]), sep="")
    main_path = opt["training_path"] + "/main.py"
    train_path = opt["training_path"] + "/pro/Train.py"
    os.system("cp -f " + main_path + " " + opt["log_file"] + "/")
    os.system("cp -f " + train_path + " " + opt["log_file"] + "/")
    print("+++++++++++++++++++++++++++++++++++++++++++")

    nlsa_encoder(opt)


def nlsa_encoder(opt):
    print("Loading training data...")
    train_data = SpadDataset(opt["train_file"], opt["train_target"], opt["noise_idx"], opt["train_size"])
    train_loader = DataLoader(train_data, batch_size=opt["batch_size"],
                              shuffle=True, num_workers=opt["workers"],
                              pin_memory=False)
    print("Load training data complete!\nLoading validation data...")
    val_data = SpadDataset(opt["val_file"], opt["train_target"], opt["noise_idx"], opt["val_size"])
    val_loader = DataLoader(val_data, batch_size=opt["batch_size"],
                            shuffle=True, num_workers=opt["workers"],
                            pin_memory=False)
    print("Load validation data complete!")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Constructing Models...")
    print("Use_network: {}\n".format(opt["use_network"]),
          "Lrg: {}\n".format(opt["lrg"]),
          "lrg_update: {}\n".format(opt["lrg_update"]),
          "Optimizer: {}\n".format(opt["optimizer"]),
          "Model_name: {}".format(opt["model_name"]), sep="")
    print("+++++++++++++++++++++++++++++++++++++++++++")
    logWriter = SummaryWriter(opt["log_file"] + "/")
    n_iter = -1
    model_g = model_NLS_Encoder.NLSA_Encoder()
    model_g.cuda()  # 使用GPU
    params = filter(lambda p: p.requires_grad, model_g.parameters())
    if opt["load_save_model_flag"] == 1 and opt["train_target"] != 2:
        checkpoint = torch.load(opt["load_save_model_path"])
        model_g.load_state_dict(checkpoint["state_dict"])
        opt["start_epoch"] = checkpoint["epoch"] + 1
        n_iter = checkpoint["n_iter"]
        opt['lrg'] = checkpoint["lr"] * opt["lrg_update"] * 2
        print("load pretrain complete !")
    optimizer_g = torch.optim.Adam(params, opt['lrg'])
    val_loss = {"RMSE": []}
    train_loss = {"loss_g": [], "loss_0": [], "loss_1": []}  # 定义train_loss的结构
    for epoch in range(opt["start_epoch"], opt["epoch"]):
        try:
            t_s = time.time()
            print("=========================>Train Start<=============================")
            print("Epoch: {}, LR_g: {}".format(epoch, optimizer_g.param_groups[0]["lr"]))
            # 第一行获得train后的返回值，tain为训练代码，在pro/Train中
            Mod_Dict_g, optimizer_g, n_iter, train_loss_epoch, val_loss, logWriter = \
                Train.train_nlsa_encoder(model_g, train_loader, val_loader, optimizer_g, epoch, n_iter, val_loss, opt,
                                   logWriter)
            train_loss_temp = [np.mean(train_loss_epoch["loss_g"][-(len(train_data) - 1):]),
                               np.mean(train_loss_epoch["loss_0"][-(len(train_data) - 1):]),
                               np.mean(train_loss_epoch["loss_1"][-(len(train_data) - 1):])]
            train_loss["loss_g"].append(train_loss_temp[0])
            train_loss["loss_0"].append(train_loss_temp[1])
            train_loss["loss_1"].append(train_loss_temp[2])
            # 显示train和validation的KL，TV，RMSE，为nan时说明出错
            print("=========================>Train Loss<==============================")
            print("loss_g: {}, MAE: {}, RMSE: {}".format(
                train_loss_temp[0], train_loss_temp[1], train_loss_temp[2]))
            # lr update
            for g in optimizer_g.param_groups:
                g["lr"] *= opt["lrg_update"]
            # save checkpoint every epoch 生成chekpoint文件，即为output中的epochxxx.pth
            scio.savemat(file_name=opt["log_file"] + "/train_loss.mat", mdict=train_loss)
            t_e = time.time()
            print("End of epoch: {}.  ".format(epoch) + "run time: %.2f min" % ((t_e - t_s) / 60))
        except RuntimeError as oom:
            if 'out of memory' in str(oom):
                print('| WARNING: ran out of memory！\n| WARNING: ran out of memory！')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise oom


if __name__ == "__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Execuating code...")
    main()
