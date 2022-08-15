# The parse argments 存放config.ini中数据
import os
import sys
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime


def parse_args(config_path="./config.ini"):
    if os.path.exists(config_path):
        print("Reading config file from {} and parse args...".format(config_path))
        opt = {}
        config = ConfigParser(interpolation=ExtendedInterpolation())  # use ConfigParser realize a instance
        config.read(config_path)  # read the config file
        config_bk = ConfigParser()

        opt["use_network"] = config.getint("params", "use_network")

        opt["use_gpu"] = config.getboolean("params", "use_gpu")
        opt["workers"] = config.getint("params", "workers")
        opt["all_test_flag"] = config.getboolean("params", "all_test_flag")
        opt["train_target"] = config.getint("params", "train_target")
        opt["load_save_model_flag"] = config.getint("params", "load_save_model_flag")
        opt["train_size"] = config.getint("params", "train_size")
        opt["val_size"] = config.getint("params", "val_size")

        opt["epoch"] = config.getint("params", "epoch")
        opt["save_every"] = config.getint("params", "save_every")
        opt["start_save_epoch"] = config.getint("params", "start_save_epoch")
        opt["start_test_epoch"] = config.getint("params", "start_test_epoch")
        opt["start_epoch"] = config.getint("params", "start_epoch")
        opt["noise_idx"] = config.getint("params", "noise_idx")
        opt["training_path"] = config.get("params", "training_path")
        opt["log_dir"] = config.get("params", "log_dir")
        opt["util_dir"] = opt["training_path"] + "/" + config.get("params", "util_dir")
        opt["load_save_model_path0"] = config.get("params", "load_save_model_path0")
        opt["load_save_model_path1"] = config.get("params", "load_save_model_path1")
        if opt["train_target"] == 0:
            opt["load_save_model_path"] = config.get("params", "load_save_model_path0")
        else:
            opt["load_save_model_path"] = config.get("params", "load_save_model_path1")

        if opt["all_test_flag"]:
            opt["train_file"] = opt["training_path"] + "/" + config.get("params", "train_file_test")
            opt["val_file"] = opt["training_path"] + "/" + config.get("params", "val_file_test")
        else:
            opt["train_file"] = opt["training_path"] + "/" + config.get("params", "train_file")
            opt["val_file"] = opt["training_path"] + "/" + config.get("params", "val_file")

        opt["test_dist_file"] = opt["training_path"] + "/" + config.get("params", "test_dist_file")

        if 0 <= opt["use_network"] < 10:
            opt["batch_size"] = config.getint("params", "batch_size00")
            opt["optimizer"] = config.get("params", "optimizer00")
            opt["lrg"] = config.getfloat("params", "lrg00")
            opt["lrg_update"] = config.getfloat("params", "lrg_update00")
            opt["p_tv"] = config.getfloat("params", "p_tv00")
            opt["model_name"] = config.get("params", "model_name00")
            opt["log_file"] = opt["log_dir"] + opt["model_name"] + "/" + datetime.now().strftime("%m_%d-%H_%M") \
                              + "-" + str(opt["use_network"]) + "-" + str(opt["train_target"])
        if 10 <= opt["use_network"] < 20:
            opt["batch_size"] = config.getint("params", "batch_size10")
            opt["optimizer"] = config.get("params", "optimizer10")
            opt["lrg"] = config.getfloat("params", "lrg10")
            opt["lrg_update"] = config.getfloat("params", "lrg_update10")
            opt["p_a"] = config.getfloat("params", "p_a10")
            opt["model_name"] = config.get("params", "model_name10")
            opt["log_file"] = opt["log_dir"] + opt["model_name"] + "/" + datetime.now().strftime("%m_%d-%H_%M") \
                              + "-" + str(opt["use_network"]) + "-" + str(opt["train_target"])
        if 20 <= opt["use_network"] < 30:
            opt["batch_size"] = config.getint("params", "batch_size20")
            opt["optimizer"] = config.get("params", "optimizer20")
            opt["lrg"] = config.getfloat("params", "lrg20")
            opt["lrg_update"] = config.getfloat("params", "lrg_update20")
            opt["p_a"] = config.getfloat("params", "p_a20")
            opt["model_name"] = config.get("params", "model_name20")
            opt["log_file"] = opt["log_dir"] + opt["model_name"] + "/" + datetime.now().strftime("%m_%d-%H_%M") \
                              + "-" + str(opt["use_network"]) + "-" + str(opt["train_target"])

        opt["save_every"] = opt["save_every"] // opt["batch_size"]

        config_bk.read_dict({"params_bk": opt})
        config_bk_pth = opt["log_file"] + "/config_bk.ini"
        if not os.path.exists(opt["log_file"]):
            os.makedirs(opt["log_file"])
        with open(config_bk_pth, "w") as cbk_pth:
            config_bk.write(cbk_pth)

        print("Config file load complete! \nNew file saved to {}".format(config_bk_pth))
        return opt
    else:
        print("No file exist named {}".format(config_path))
        sys.exit("NO FILE ERROR")
