import numpy as np
import torch
from glob import glob
import scipy
import os
import scipy.io as scio
import time
import tqdm
# from pro.Loss import criterion_KL, criterion_tv, criterion_l2
import cv2
# from pro.Loss import criterion_L2

dtype = torch.cuda.FloatTensor


# test function for Middlebury dataset
def test_sm(model, opt, outdir_m):
    rmse_all = []
    time_all = []

    for name_test in glob(opt["testDataDir"] + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        print("Loading data {} and processing...".format(name_test_id))

        dep = scio.loadmat(name_test)["depth"]
        dep = np.asarray(dep).astype(np.float32)
        h, w = dep.shape

        M_mea = scio.loadmat(name_test)["spad"]
        M_mea = scipy.sparse.csc_matrix.todense(M_mea)
        M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
        M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 2, 3))).type(dtype)

        t_s = time.time()
        M_mea_re, dep_re = model(M_mea)
        t_e = time.time()
        time_all.append(t_e - t_s)

        C = 3e8
        Tp = 100e-9

        dist = dep_re.data.cpu().numpy()[0, 0, :, :] * Tp * C / 2
        rmse = np.sqrt(np.mean((dist - dep) ** 2))
        rmse_all.append(rmse)

        scio.savemat(name_test_save, {"data": dist, "rmse": rmse})
        print("The RMSE: {}".format(rmse))

    return np.mean(rmse_all), np.mean(time_all)


def test_t1(model, datadir, outdir_m):
    rmse_all = []
    time_all = []

    for name_test in glob(datadir + "/" + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        # print("Loading data {} and processing...".format(name_test_id))

        dep = (np.asarray(scipy.io.loadmat(name_test)['bin']).astype(
            np.float32).reshape([64, 64]) - 1)[:, :] / 1023
        h, w = dep.shape
        M_mea = scio.loadmat(name_test)["spad"]
        M_mea = scipy.sparse.csc_matrix.todense(M_mea)
        M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
        M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 2, 3))).type(dtype)

        t_s = time.time()
        M_mea_re, dep_re = model(M_mea)
        t_e = time.time()
        time_all.append(t_e - t_s)

        C = 1
        Tp = 2

        dist = dep_re.data.cpu().numpy()[0, 0, :, :] * Tp * C / 2
        for i in range(len(dist)):
            for j in range(i, len(dist)):
                tmp = dist[i, j]
                dist[i, j] = dist[j, i]
                dist[j, i] = tmp
        rmse = np.sqrt(np.mean((dist - dep) ** 2)) * 12.276
        rmse_all.append(rmse)

        scio.savemat(name_test_save, {"data": dist, "rmse": rmse})
        # print("The RMSE: {}".format(rmse))

    return np.mean(rmse_all), np.mean(time_all)


def test_t2(model_g, model_d, datadir, outdir_m):
    rmse_all = []
    time_all = []

    for name_test in glob(datadir + "/" + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        # print("Loading data {} and processing...".format(name_test_id))

        dep = (np.asarray(scipy.io.loadmat(name_test)['bin']).astype(
            np.float32).reshape([64, 64]) - 1)[:, :] / 1023
        h, w = dep.shape
        M_mea = scio.loadmat(name_test)["spad"]
        M_mea = scipy.sparse.csc_matrix.todense(M_mea)
        M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
        M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 3, 2))).type(dtype)

        t_s = time.time()
        M_mea_re, dep_re = model_g(M_mea)
        image_out = model_d(dep_re)
        t_e = time.time()
        time_all.append(t_e - t_s)

        C = 1
        Tp = 2
        dep_re = dep_re.data.cpu().numpy()[0, 0, :, :]
        dist = image_out.data.cpu().numpy()[0, 0, :, :] * Tp * C / 2
        rmse = np.sqrt(np.mean((dist - dep) ** 2)) * 12.276
        rmse_g = np.sqrt(np.mean((dep_re - dep) ** 2)) * 12.276
        rmse_all.append(rmse)

        scio.savemat(name_test_save, {"dep_re": dep_re, "data": dist, "rmse_g": rmse_g, "rmse": rmse})
        # print("The RMSE: {}".format(rmse))

    return np.mean(rmse_all), np.mean(time_all)


def test_t3(model, datadir, outdir_m):
    rmse_all = []
    time_all = []

    for name_test in glob(datadir + "/" + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        # print("Loading data {} and processing...".format(name_test_id))

        dep = (np.asarray(scipy.io.loadmat(name_test)['bin']).astype(
            np.float32).reshape([64, 64]) - 1)[:, :] / 1023
        h, w = dep.shape
        M_mea = scio.loadmat(name_test)["spad"]
        M_mea = scipy.sparse.csc_matrix.todense(M_mea)
        M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, h, w, -1])
        M_mea = torch.from_numpy(np.transpose(M_mea, (0, 1, 4, 2, 3))).type(dtype)

        t_s = time.time()
        M_mea_re, dep_re = model(M_mea)
        t_e = time.time()
        time_all.append(t_e - t_s)

        C = 1
        Tp = 2

        dist = dep_re.data.cpu().numpy()[0, 0, :, :] * Tp * C / 2
        for i in range(len(dist)):
            for j in range(i, len(dist)):
                tmp = dist[i, j]
                dist[i, j] = dist[j, i]
                dist[j, i] = tmp
        rmse = np.sqrt(np.mean((dist - dep) ** 2)) * 12.276
        rmse_all.append(rmse)

        scio.savemat(name_test_save, {"data": dist, "rmse": rmse})
        # print("The RMSE: {}".format(rmse))

    return np.mean(rmse_all), np.mean(time_all)


# test function for outdoor real-world dataset
def test_outrw(model, opt, outdir_m):
    rmse_all = [0, 0]
    time_all = []
    base_pad = 16
    step = 32
    grab = 32
    dim = 64

    for name_test in glob(opt["testDataDir"] + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        print("Loading data {} and processing...".format(name_test_id))
        M_mea_raw = np.asarray(scipy.io.loadmat(name_test)['y'])

        tp_s = 4770
        t_inter = 1024
        M_mea = M_mea_raw[:, :, tp_s:tp_s + t_inter]
        M_mea = M_mea.transpose((2, 0, 1))
        M_mea = torch.from_numpy(M_mea).unsqueeze(0).unsqueeze(0).type(dtype)

        out = np.zeros((M_mea.shape[1], M_mea.shape[2]))
        M_mea = torch.nn.functional.pad(M_mea, (base_pad, 0, base_pad, 0, 0, 0))  # pad only on edge

        t_s = time.clock()
        for i in tqdm(range(4)):
            for j in range(4):
                M_mea_input = M_mea[:, :, :, i * step:(i) * step + dim, j * step:(j) * step + dim]
                print("Size of input:{}".format(M_mea_input.shape))
                M_mea_re, dep_re = model(M_mea_input)
                M_mea_re = M_mea_re.data.cpu().numpy().squeeze()
                tile_out = np.argmax(M_mea_re, axis=0)
                if i <= 14:
                    out[i * step:(i + 1) * step, j * step:(j + 1) * step] = tile_out[step:step + step, step:step + step]
                else:
                    out[i * step:(i + 1) * step, j * step:(j + 1) * step] = tile_out[16:16 + step, 16:16 + step]

        t_e = time.clock()
        time_all.append(t_e - t_s)

        dist = out.astype(np.float32) * 0.15

        scio.savemat(name_test_save, {"data": dist})

    return np.mean(rmse_all), np.mean(time_all)


# test function for indoor real-world data
def test_inrw(model, opt, outdir_m):
    rmse_all = [0, 0]
    time_all = []
    base_pad = 16
    step = 16
    grab = 32
    dim = 64

    for name_test in glob(opt["testDataDir"] + "*.mat"):
        name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
        name_test_save = outdir_m + "/" + name_test_id + "_rec.mat"

        print("Loading data {} and processing...".format(name_test_id))
        M_mea = scio.loadmat(name_test)["spad_processed_data"]
        M_mea = scipy.sparse.csc_matrix.todense(M_mea)
        M_mea = np.asarray(M_mea).astype(np.float32).reshape([1, 1, 1536, 256, 256])
        M_mea = M_mea.transpose((0, 1, 2, 4, 3)).type(dtype)

        out = np.zeros((M_mea.shape[3], M_mea.shape[4]))
        M_mea = torch.nn.functional.pad(M_mea, (base_pad, 0, base_pad, 0, 0, 0))

        t_s = time.time()
        for i in tqdm(range(16)):
            for j in range(16):
                M_mea_input = M_mea[:, :, :, i * step:(i) * step + dim, j * step:(j) * step + dim]
                M_mea_re, dep_re = model(M_mea_input)
                M_mea_re = M_mea_re.data.cpu().numpy().squeeze()
                tile_out = np.argmax(M_mea_re, axis=0)
                out[i * step:(i + 1) * step, j * step:(j + 1) * step] = \
                    tile_out[step // 2:step // 2 + step, step // 2:step // 2 + step]

        t_e = time.time()
        time_all.append(t_e - t_s)

        dist = out * 6 / 1536.

        scio.savemat(name_test_save, {"data": dist})

    return np.mean(rmse_all), np.mean(time_all)


def test_tr(model, datadir, outdir_m):
    rebuild_time = []
    testlog = {"file_name": []}
    for file_test in glob(datadir + "/*"):
        ta_s = time.time()
        file_test_id, _ = os.path.splitext(os.path.split(file_test)[1])
        outpath = outdir_m + "/" + file_test_id
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for name_test in glob(file_test + "/" + "*.mat"):
            name_test_id, _ = os.path.splitext(os.path.split(name_test)[1])
            name_test_save = outpath + "/" + name_test_id + "_rec.mat"
            h, w = 64, 64
            M_mea = scio.loadmat(name_test)["spad"]
            time_range = np.size(M_mea, 1)
            if time_range > 1024:
                print("time_d out of range")
                break
            M_mea = scipy.sparse.csc_matrix.todense(M_mea)
            time_c = 1024 // time_range
            if time_c > 1:
                M_mea_re = np.zeros((4096, 1024))
                for i in range(4096):
                    for j in range(time_range):
                        M_mea_re[i, j*time_c] = M_mea[i, j]
            else:
                M_mea_re = M_mea
            M_mea_re = np.asarray(M_mea_re).astype(np.float32).reshape([1, 1, h, w, -1])
            M_mea_re = torch.from_numpy(np.transpose(M_mea_re, (0, 1, 4, 2, 3))).type(dtype)

            t_s = time.time()
            dep_re, _ = model(M_mea_re)
            t_e = time.time()
            rebuild_time.append(t_e - t_s)
            if time_c > 1:
                dep_re = dep_re.data.cpu().numpy()[0, 0, :, :] / time_c
            else:
                dep_re = dep_re.data.cpu().numpy()[0, 0, :, :]
            scio.savemat(name_test_save, {"data": dep_re})

        testlog["file_name"].append(file_test_id)
        ta_e = time.time()
        print(str((ta_e-ta_s)/60))
    scio.savemat(file_name=outdir_m + "/test_time_" + str(time.time()) + ".mat", mdict=testlog)

    return np.mean(rebuild_time)
