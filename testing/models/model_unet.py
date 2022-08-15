import torch
import torch.nn as nn
# import torch_geometric
from torch.autograd import Variable


class Pool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Pool, self).__init__()
        self.maxpool_conv = nn.MaxPool3d(2)
        self.avgpool_conv = nn.AvgPool3d(2)
        self.convout = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = torch.cat([self.maxpool_conv(x), self.avgpool_conv(x)], dim=1)
        out = self.convout(x1)
        return out


class MPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MPool, self).__init__()
        self.maxpool_conv = nn.MaxPool3d(2)
        # self.avgpool_conv = nn.AvgPool3d(2)
        self.convout = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.maxpool_conv(x)
        out = self.convout(x1)
        return out


class Dcat(nn.Module):
    def __init__(self, in_channels, de_channels, out_channels):
        super(Dcat, self).__init__()
        self.Deconv = nn.ConvTranspose3d(de_channels, de_channels, 2, 2, bias=False)
        self.convout = nn.Sequential(
            nn.Conv3d(in_channels + de_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, g):
        out1 = torch.cat([x, self.Deconv(g)], dim=1)
        out = self.convout(out1)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_channels // ratio, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.mc = ChannelAttention(in_channels)
        self.ms = SpatialAttention()

    def forward(self, x):
        # x1 = torch.mul(x, self.mc(x))
        # x2 = torch.mul(x1, self.ms(x1))
        x1 = self.mc(x)*x
        x2 = self.ms(x1)*x1
        return x + x2


class ADAG(nn.Module):
    def __init__(self, in_channels, de_channels, out_channels):
        super(ADAG, self).__init__()
        self.Deconv = nn.ConvTranspose3d(de_channels, in_channels, 2, 2, bias=False)
        self.cbam = CBAM(in_channels)
        self.convx = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.convg = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.convs = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.convout = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, g):
        g1 = self.cbam(self.Deconv(g))
        x1 = self.convx(x) + self.convg(g1)
        x2 = self.relu(x1)
        x3 = self.convs(x2)
        x4 = self.sigmoid(x3)
        x5 = torch.mul(x, x4)
        x6 = torch.cat([g1, x5], dim=1)
        out = self.convout(x6)
        return out


class UNetADAG(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetADAG, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU()
        )
        self.pool10 = Pool(4, 4)
        self.pool20 = Pool(4, 8)
        self.pool30 = Pool(8, 12)
        self.adag01 = ADAG(4, 4, 6)
        self.dcat11 = Dcat(4, 8, 12)
        self.dcat21 = Dcat(8, 12, 10)
        self.adag02 = ADAG(10, 12, 11)
        self.dcat12 = Dcat(4 + 12, 10, 12)
        self.adag03 = ADAG(4 + 6 + 11, 12, 10)
        self.cbam = CBAM(10)
        self.convout = nn.Conv3d(10, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        smax = torch.nn.Softmax2d()
        x00 = self.inc(x)
        x10 = self.pool10(x00)
        x20 = self.pool20(x10)
        x30 = self.pool30(x20)
        x01 = self.adag01(x00, x10)
        x11 = self.dcat11(x10, x20)
        x21 = self.dcat21(x20, x30)
        x02 = self.adag02(torch.cat([x00, x01], dim=1), x11)
        x12 = self.dcat12(torch.cat([x10, x11], dim=1), x21)
        x03 = self.adag03(torch.cat([x00, x01, x02], dim=1), x12)
        re1 = self.cbam(x03)
        convrout = self.convout(re1)
        denoise_out = torch.squeeze(convrout, 1)

        weights = Variable(
            torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax(denoise_out)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return soft_argmax, denoise_out


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            nn.Conv3d(in_channels, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(4),
            nn.ReLU()
        )
        self.pool10 = MPool(4, 4)
        self.pool20 = MPool(4, 8)
        self.pool30 = MPool(8, 12)
        # self.adag01 = ADAG(4, 4, 6)
        self.dcat01 = Dcat(4, 4, 6)
        self.dcat11 = Dcat(4, 8, 12)
        self.dcat21 = Dcat(8, 12, 10)
        # self.adag02 = ADAG(10, 12, 11)
        self.dcat02 = Dcat(4 + 6, 12, 11)
        self.dcat12 = Dcat(4 + 12, 10, 12)
        # self.adag03 = ADAG(4 + 6 + 11, 12, 10)
        self.dcat03 = Dcat(4 + 6 + 11, 12, 10)
        # self.cbam = CBAM(10)
        self.convout = nn.Conv3d(10, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        smax = torch.nn.Softmax2d()
        x00 = self.inc(x)
        x10 = self.pool10(x00)
        x20 = self.pool20(x10)
        x30 = self.pool30(x20)
        # x01 = self.adag01(x00, x10)
        x01 = self.dcat01(x00, x10)
        x11 = self.dcat11(x10, x20)
        x21 = self.dcat21(x20, x30)
        # x02 = self.adag02(torch.cat([x00, x01], dim=1), x11)
        x02 = self.dcat02(torch.cat([x00, x01], dim=1), x11)
        x12 = self.dcat12(torch.cat([x10, x11], dim=1), x21)
        # x03 = self.adag03(torch.cat([x00, x01, x02], dim=1), x12)
        x03 = self.dcat03(torch.cat([x00, x01, x02], dim=1), x12)
        # re1 = self.cbam(x03)
        convrout = self.convout(x03)
        denoise_out = torch.squeeze(convrout, 1)

        weights = Variable(
            torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax(denoise_out)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return soft_argmax, denoise_out
