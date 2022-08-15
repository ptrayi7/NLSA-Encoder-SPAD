# This is the file for loss functions
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False, weights=None):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3):
        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3, weights=None):
        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range,
                       weights=self.weights)


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    # grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    # the computed grad's edge is useless, remove them
    grad_x = grad_x[:, :, 2:-2, 2:-2]
    grad_y = grad_y[:, :, 2:-2, 2:-2]
    out = torch.cat((torch.reshape(grad_y, (N, C, -1)), torch.reshape(grad_x, (N, C, -1))), dim=1)
    return out


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    def forward(self, inpt, tar):
        grad_tar = imgrad_yx(tar)
        grad_inpt = imgrad_yx(inpt)
        loss = torch.sum(torch.mean(torch.abs(grad_tar - grad_inpt)))
        return loss


class L1_log(nn.Module):
    def __init__(self):
        super(L1_log, self).__init__()

    def forward(self, inpt, target):
        loss = nn.L1Loss()(torch.log(inpt), torch.log(target))
        return loss


###########################################
# the set of loss functions
criterion_GAN = nn.MSELoss()
criterion_KL = nn.KLDivLoss()

# inpt, target: [batch_size, 1, h, w]
criterion_L1 = nn.L1Loss()
criterion_L1log = L1_log()
criterion_grad = GradLoss()


def criterion_tv(inpt):
    return torch.sum(torch.abs(inpt[:, :, :, :-1] - inpt[:, :, :, 1:])) + \
           torch.sum(torch.abs(inpt[:, :, :-1, :] - inpt[:, :, 1:, :]))


def criterion_l2(est, gt):
    criterion = nn.MSELoss()
    # est should have grad
    return torch.sqrt(criterion(est, gt))


# def edge_conv2d(im):
#     # 用nn.Conv2d定义卷积操作
#     conv_op = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#     # 定义sobel算子参数，所有值除以3个人觉得出来的图更好些
#     sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
#     # 将sobel算子转换为适配卷积操作的卷积核
#     sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
#     # 卷积输出通道，这里我设置为3
#     sobel_kernel = np.repeat(sobel_kernel, 1, axis=1)
#     # 输入图的通道，这里我设置为3
#     sobel_kernel = np.repeat(sobel_kernel, 1, axis=0)
#
#     conv_op.weight.data = torch.from_numpy(sobel_kernel)
#     # print(conv_op.weight.size())
#     # print(conv_op, '\n')
#
#     edge_detect = conv_op(im)
#     print(torch.max(edge_detect))
#     # 将输出转换为图片格式
#     edge_detect = edge_detect.squeeze().detach().numpy()
#     return edge_detect
#
#
# def sobel_grad(img):
#     edge_detect = edge_conv2d(img)
#     edge_detect = np.transpose(edge_detect, (1, 2, 0))
#     # cv2.imshow('edge.jpg', edge_detect)
#     # cv2.waitKey(0)
#     cv2.imwrite('edge-2.jpg', edge_detect)
#
#
# def criterion_gr(est, gt):
#     criterion = nn.MSELoss()
#     # est should have grad
#     return torch.sqrt(criterion(sobel_grad(est), sobel_grad(gt)))


def criterion_l1(est, gt):
    criterion = nn.L1Loss()
    # est should have grad
    return criterion(est, gt)
