# The train, validation list make function
from glob import glob
import re
import os.path
import sys

simulation_param_idx = 1  # 1 or 2 corresponding to that in SimulateTrainMeasurements.m
project_folder = os.path.abspath('/home/ptrayi/projecct/SP') + '/'  # 项目根目录
dataset_folder = os.path.abspath(
    '/home/ptrayi/dataset/2_100') + '/'  # 存放的数据集目录


def intersect_files(train_files):
    intensity_train_files = []
    for t in train_files:
        intensity_train_files.append(glob(dataset_folder + t + 'intensity*.mat'))
    intensity_train_files = [file for sublist in intensity_train_files for file in sublist]

    spad_train_files = []
    if simulation_param_idx is not None:
        noise_param = [simulation_param_idx]
    else:
        print("Specify simulation_param_idx = 1 or 2")
        sys.exit("SIMULATION PARAMETER INDEX ERROR")

    for p in noise_param:
        spad_train_files.append([])
        for t in train_files:
            spad_train_files[-1].append(glob(dataset_folder + t + 'spad*p{}.mat'.format(p)))
        spad_train_files[-1] = [file for sublist in spad_train_files[-1] for file in sublist]

        spad_train_files[-1] = [re.sub(r'(.*)/spad_(.*)_p.*.mat', r'\1/intensity_\2.mat',
                                       file) for file in spad_train_files[-1]]

    intensity_train_files = set(intensity_train_files)

    for idx, p in enumerate(noise_param):
        spad_train_files[idx] = set(spad_train_files[idx])
    intensity_train_files = intensity_train_files.intersection(*tuple(
        spad_train_files))
    # 检测intensity是否有对应的spad文件ad
    return intensity_train_files


def main():
    with open(project_folder + 'training/util/train2_100.txt') as f:  # 需要人工输入train.txt内容
        train_files = f.read().split()
    with open(project_folder + 'training/util/val.txt') as f:  # 需要人工输入val.txt内容
        val_files = f.read().split()

    print('Sorting training files')
    intensity_train_files = intersect_files(train_files)  # 进入函数获取清单
    print('Sorting validation files')
    intensity_val_files = intersect_files(val_files)  # 进入函数获取清单

    print('Writing training files')
    with open(project_folder + 'training/util/train2_100_intensity.txt', 'w') as f:  # 将清单写入文件
        for file in intensity_train_files:
            f.write(file + '\n')
    print('Writing validation files')
    # with open(project_folder + 'training/util/val_intensity.txt', 'w') as f:  # 将清单写入文件
    #     for file in intensity_val_files:
    #         f.write(file + '\n')

    print('Wrote {} train, {} validation files'.format(len(intensity_train_files),
                                                       len(intensity_val_files)))
    return


if __name__ == '__main__':
    main()
