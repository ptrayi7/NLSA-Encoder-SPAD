import torch
import torch.nn as nn
from models import model_DRNLSA, model_NLSA, model_block


class Shortblock(nn.Module):
    def __init__(self, in_channels=64):
        super(Shortblock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        resblock_out = self.resblock(inputs)
        return resblock_out


class ResnetEncoderblock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=0):
        super(ResnetEncoderblock, self).__init__()
        if (in_channels // 4) < 64:
            temp_channels = 64
        else:
            temp_channels = in_channels // 4
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, temp_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_channels, temp_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(temp_channels, out_channels, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, inputs):
        resblock_out = self.resblock(inputs)
        return resblock_out


class DimensionReuctionNLSA_BN(nn.Module):
    def __init__(self):
        super(DimensionReuctionNLSA_BN, self).__init__()
        self.drnlsa = model_DRNLSA.DRNLSA(1024, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inputsd = torch.squeeze(inputs, 1)

        drnlsaout = self.drnlsa(inputsd)
        restructureout = self.relu(drnlsaout)

        return restructureout, inputsd


class DimensionReuctionNLSA_BRN(nn.Module):
    def __init__(self):
        super(DimensionReuctionNLSA_BRN, self).__init__()
        self.drnlsa = model_DRNLSA.DRNLSA(1024, 1, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inputsd = torch.squeeze(inputs, 1)

        drnlsaout = self.drnlsa(inputsd)
        restructureout = self.relu(drnlsaout)

        return restructureout, inputsd


class DRNLSANR(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1):
        super(DRNLSANR, self).__init__()
        self.nlsa1024 = model_NLSA.NonLocalSparseAttention(in_channels)
        self.resblock1024_0 = Shortblock(in_channels)
        self.resblock1024_1 = Shortblock(in_channels)
        self.resblock1024_2 = Shortblock(in_channels)
        self.resjumpblock1024d = model_block.ResnetEncoderblock(in_channels, in_channels // 2)
        the_channels = in_channels // 2

        self.nlsa512 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock512_0 = Shortblock(the_channels)
        self.resblock512_1 = Shortblock(the_channels)
        self.resblock512_2 = Shortblock(the_channels)
        self.resjumpblock512d = model_block.ResnetEncoderblock(the_channels, the_channels // 2)
        the_channels = the_channels // 2

        self.nlsa256 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock256_0 = Shortblock(the_channels)
        self.resblock256_1 = Shortblock(the_channels)
        self.resblock256_2 = Shortblock(the_channels)
        self.resjumpblock256d = model_block.ResnetEncoderblock(the_channels, the_channels // 2)
        the_channels = the_channels // 2

        self.nlsa128 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock128_0 = Shortblock(the_channels)
        self.resblock128_1 = Shortblock(the_channels)
        self.resblock128_2 = Shortblock(the_channels)
        self.resjumpblock128d = model_block.ResnetEncoderblock(the_channels, the_channels // 2)
        the_channels = the_channels // 2

        self.nlsa64 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock64_0 = Shortblock(the_channels)
        self.resblock64_1 = Shortblock(the_channels)
        self.resblock64_2 = Shortblock(the_channels)

        self.C0 = nn.Conv2d(the_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, inputs):
        ns1024 = self.nlsa1024(inputs)
        r1024 = self.resblock1024_0(ns1024)
        r1024 = self.resblock1024_1(r1024)
        r1024 = self.resblock1024_2(r1024)
        rj1024d = self.resjumpblock1024d(r1024)

        ns512 = self.nlsa512(rj1024d)
        r512 = self.resblock512_0(ns512)
        r512 = self.resblock512_1(r512)
        r512 = self.resblock512_2(r512)
        rj512d = self.resjumpblock512d(r512)

        ns256 = self.nlsa256(rj512d)
        r256 = self.resblock256_0(ns256)
        r256 = self.resblock256_1(r256)
        r256 = self.resblock256_2(r256)
        rj256d = self.resjumpblock256d(r256)

        ns128 = self.nlsa128(rj256d)
        r128 = self.resblock128_0(ns128)
        r128 = self.resblock128_1(r128)
        r128 = self.resblock128_2(r128)
        rj128d = self.resjumpblock128d(r128)

        ns64 = self.nlsa64(rj128d)
        r64 = self.resblock64_0(ns64)
        r64 = self.resblock64_1(r64)
        r64 = self.resblock64_2(r64)
        out = self.C0(r64)

        return out


class DimensionReuctionNLSA_nres(nn.Module):
    def __init__(self):
        super(DimensionReuctionNLSA_nres, self).__init__()
        self.drnlsa = DRNLSANR(1024, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inputsd = torch.squeeze(inputs, 1)

        drnlsaout = self.drnlsa(inputsd)
        restructureout = self.relu(drnlsaout)

        return restructureout, inputsd


class DimensionReuctionNLSA_nNLSA(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1, bn=2):
        super(DimensionReuctionNLSA_nNLSA, self).__init__()
        self.bnmodel = bn
        self.resblock1024_0 = model_block.Resnetlongblock(in_channels, bn=self.bnmodel)
        self.resblock1024_1 = model_block.Resnetlongblock(in_channels, bn=self.bnmodel)
        self.resblock1024_2 = model_block.Resnetlongblock(in_channels, bn=self.bnmodel)
        self.resjumpblock1024d = model_block.ResnetEncoderblock(in_channels, in_channels // 2, bn=self.bnmodel)
        the_channels = in_channels // 2

        self.resblock512_0 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock512_1 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock512_2 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resjumpblock512d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.resblock256_0 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock256_1 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock256_2 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resjumpblock256d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.resblock128_0 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock128_1 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock128_2 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resjumpblock128d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.resblock64_0 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock64_1 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock64_2 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)

        self.C0 = nn.Conv2d(the_channels, out_channels, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inputsd = torch.squeeze(inputs, 1)

        r1024_0 = self.resblock1024_0(inputsd)
        r1024_1 = self.resblock1024_1(r1024_0)
        r1024_2 = self.resblock1024_2(r1024_1)
        rj1024d = self.resjumpblock1024d(r1024_2)

        r512_0 = self.resblock512_0(rj1024d)
        r512_1 = self.resblock512_1(r512_0)
        r512_2 = self.resblock512_2(r512_1)
        rj512d = self.resjumpblock512d(r512_2)

        r256_0 = self.resblock256_0(rj512d)
        r256_1 = self.resblock256_1(r256_0)
        r256_2 = self.resblock256_2(r256_1)
        rj256d = self.resjumpblock256d(r256_2)

        r128_0 = self.resblock128_0(rj256d)
        r128_1 = self.resblock128_1(r128_0)
        r128_2 = self.resblock128_2(r128_1)
        rj128d = self.resjumpblock128d(r128_2)

        r64_0 = self.resblock64_0(rj128d)
        r64_1 = self.resblock64_1(r64_0)
        r64_2 = self.resblock64_2(r64_1)
        out = self.C0(r64_2)
        output = self.relu(out)

        return output, inputsd
