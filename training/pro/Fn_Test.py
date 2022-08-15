import numpy as np
import torch
from glob import glob
import scipy
import os
import scipy.io as scio
# import cv2
import time

dtype = torch.cuda.FloatTensor


def test_t(model, epoch, params, outdir_m):
    # flag_save = 0
    if not os.path.exists(outdir_m):
        os.makedirs(outdir_m)
    rmse_all = []
    time_all = []
    testlog = {"SBR": [], "RMSE": []}
    for file_test in glob(params["test_dist_file"] + "/" + "*"):
        rmse_temp = []
        for name_test in glob(file_test + "/" + "*.mat"):
            name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
            name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

            # print("Loading data {} and processing...".format(name_test_id))

            if params["train_target"] == 0:
                dep = np.asarray(scipy.io.loadmat(name_test)['depth']).astype(
                    np.float32)[:, :]
                h, w = dep.shape
            elif params["train_target"] == 1:
                dep = (np.asarray(scipy.io.loadmat(name_test)['intensity']).astype(np.float32))
                h, w = dep.shape
            else:
                dist = np.asarray(scipy.io.loadmat(name_test)['depth']).astype(
                    np.float32)[None, :, :]
                intensity = (np.asarray(scipy.io.loadmat(name_test)['intensity']).astype(np.float32))[None, :, :]
                dep = np.append(dist, intensity, axis=0)
                h = dep.shape[1]
                w = dep.shape[2]
            M_mea = scio.loadmat(name_test)["spad"]
            M_mea = scipy.sparse.csc_matrix.todense(M_mea)
            M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
            M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 3, 2))).type(dtype)

            t_s = time.time()
            dep_re, _ = model(M_mea)
            t_e = time.time()
            time_all.append(t_e - t_s)
            TP = 100e-9/1024
            C = 3e8
            if params["train_target"] == 0:
                dep_re = (dep_re.data.cpu().numpy()[0, 0, :, :] * 1023 + 1) * TP * C / 2
            elif params["train_target"] == 1:
                dep_re = dep_re.data.cpu().numpy()[0, 0, :, :]
                dep_re = inno(dep_re)
            else:
                dep_re_d = (dep_re.data.cpu().numpy()[0, 0, :, :] * 1023 + 1) * TP * C / 2
                dep_re_i = dep_re.data.cpu().numpy()[0, 1, :, :]
                dep_re = np.append(dep_re_d[None, :, :], dep_re_i[None, :, :], axis=0)
                rmse0 = np.sqrt(np.mean((dep_re_d - dep[0, :, :]) ** 2))
                rmse1 = np.sqrt(np.mean((dep_re_i - dep[1, :, :]) ** 2))
                scio.savemat(name_test_save, {"depth_re": dep_re_d, "rmse0": rmse0, "intensity_re": dep_re_i, "rmse1": rmse1})
            rmse = np.sqrt(np.mean((dep_re - dep) ** 2))
            rmse_temp.append(rmse)
            rmse_all.append(rmse)
            if params["train_target"] != 2:
                scio.savemat(name_test_save, {"data": dep_re, "rmse": rmse})
                # savebest(params["train_target"], params["training_path"], name_test_id, dep_re, rmse)

        sbrname = os.path.split(file_test)[1].split("/")[0]
        testlog["SBR"].append(sbrname)
        testlog["RMSE"].append(np.mean(rmse_temp))

        if sbrname == '2_50':
            print("2_50--RMSE: {}".format(np.mean(rmse_temp)))
        if sbrname == '2_100':
            print("2_100--RMSE: {}".format(np.mean(rmse_temp)))
        if sbrname == 'LR128':
            print("LR128--RMSE: {}".format(np.mean(rmse_temp)))
    flag_save = 1
    if flag_save == 1:
        scio.savemat(outdir_m + "/test_loss.mat", mdict=testlog)
    else:
        for name in glob(outdir_m + "/" + "*"):
            os.remove(name)
        os.removedirs(outdir_m)
    print("Test Loss--RMSE: {}".format(np.mean(rmse_all)))
    print("Test Time--RMSE: {}".format(np.mean(time_all)))
    return flag_save


def savebest(target, savepath, filename, img_new, rmse_new):
    if target == 0:
        best_path = savepath + "/best_imgout/depth/"+filename + "_rec.mat"
    else:
        best_path = savepath + "/best_imgout/intensity/" + filename + "_rec.mat"
    rmse_old = scipy.io.loadmat(best_path)['rmse']
    if rmse_new < rmse_old:
        scio.savemat(best_path, {"data": img_new, "rmse": rmse_new})
        print("save : "+str(best_path))
    return 0


def test_t2(model, model_pre, params, outdir_m):
    # flag_save = 0
    if not os.path.exists(outdir_m):
        os.makedirs(outdir_m)
    rmse_all = []
    time_all = []
    testlog = {"SBR": [], "RMSE": []}
    for file_test in glob(params["test_dist_file"] + "/" + "*"):
        rmse_temp = []
        for name_test in glob(file_test + "/" + "*.mat"):
            name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
            name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

            # print("Loading data {} and processing...".format(name_test_id))

            if params["train_target"] == 0:
                dep = np.asarray(scipy.io.loadmat(name_test)['depth']).astype(
                    np.float32)[:, :]
                h, w = dep.shape
            elif params["train_target"] == 1:
                dep = (np.asarray(scipy.io.loadmat(name_test)['intensity']).astype(np.float32))
                h, w = dep.shape
            else:
                dist = np.asarray(scipy.io.loadmat(name_test)['depth']).astype(
                    np.float32)[None, :, :]
                intensity = (np.asarray(scipy.io.loadmat(name_test)['intensity']).astype(np.float32))[None, :, :]
                dep = np.append(dist, intensity, axis=0)
                h = dep.shape[1]
                w = dep.shape[2]
            M_mea = scio.loadmat(name_test)["spad"]
            M_mea = scipy.sparse.csc_matrix.todense(M_mea)
            M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
            M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 3, 2))).type(dtype)

            t_s = time.time()
            dep_re_p, rec128 = model_pre(M_mea)
            dep_re = model(rec128)
            t_e = time.time()
            time_all.append(t_e - t_s)
            TP = 100e-9/1024
            C = 3e8
            if params["train_target"] == 0:
                # dep_re = dep_re.data.cpu().numpy()[0, 0, :, :]
                dep_re = (dep_re.data.cpu().numpy()[0, 0, :, :] * 1023 + 1) * TP * C / 2
                dep_re_p = dep_re_p.data.cpu().numpy()[0, 0, :, :]
                dep_re_p = inno(dep_re_p)
            elif params["train_target"] == 1:
                dep_re_i = dep_re.data.cpu().numpy()[0, 0, :, :]
                dep_re_p = (dep_re_p.data.cpu().numpy()[0, 0, :, :] * 1023 + 1) * TP * C / 2
                dep_re = inno(dep_re_i)
            else:
                dep_re_p = (dep_re.data.cpu().numpy()[0, 0, :, :] * 1023 + 1) * TP * C / 2
                dep_re_i = dep_re.data.cpu().numpy()[0, 1, :, :]
                dep_re = np.append(dep_re_p[None, :, :], dep_re_i[None, :, :], axis=0)
            rmse = np.sqrt(np.mean((dep_re - dep) ** 2))
            rmse_temp.append(rmse)
            rmse_all.append(rmse)
            if params["train_target"] == 0:
                scio.savemat(name_test_save, {"depth_re": dep_re, "rmse": rmse, "intensity_re": dep_re_p})
            elif params["train_target"] == 1:
                scio.savemat(name_test_save, {"intensity_re": dep_re, "rmse": rmse, "depth_re": dep_re_p})
            # savebest(params["train_target"], params["training_path"], name_test_id, dep_re, rmse)

        sbrname = os.path.split(file_test)[1].split("/")[0]
        testlog["SBR"].append(sbrname)
        testlog["RMSE"].append(np.mean(rmse_temp))

        if sbrname == '2_50':
            print("2_50--RMSE: {}".format(np.mean(rmse_temp)))
        if sbrname == '2_100':
            print("2_100--RMSE: {}".format(np.mean(rmse_temp)))
        if sbrname == 'LR128':
            print("LR128--RMSE: {}".format(np.mean(rmse_temp)))
    flag_save = 1
    if flag_save == 1:
        scio.savemat(outdir_m + "/test_loss.mat", mdict=testlog)
    else:
        for name in glob(outdir_m + "/" + "*"):
            os.remove(name)
        os.removedirs(outdir_m)
    print("Test Loss--RMSE: {}".format(np.mean(rmse_all)))
    print("Test Time--RMSE: {}".format(np.mean(time_all)))
    return flag_save


def inno(dep):
    min_i = np.min(dep)
    if min_i < 0:
        dep_0 = dep + min_i
    else:
        dep_0 = dep
    max_i = np.max(dep_0)
    if max_i > 1:
        dep_re = dep_0/max_i
    else:
        dep_re = dep_0
    return dep_re
