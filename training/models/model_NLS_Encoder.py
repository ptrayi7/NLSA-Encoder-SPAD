import torch
import torch.nn as nn
from models import model_block, model_NLSA


class DRNLSA(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1, bn=0):
        super(DRNLSA, self).__init__()
        self.bnmodel = bn
        self.nlsa1024 = model_NLSA.NonLocalSparseAttention(in_channels)
        self.resblock1024_0 = model_block.Resnetlongblock(in_channels, bn=self.bnmodel)
        self.resblock1024_1 = model_block.Resnetlongblock(in_channels, bn=self.bnmodel)
        self.resblock1024_2 = model_block.Resnetlongblock(in_channels, bn=self.bnmodel)
        self.resjumpblock1024d = model_block.ResnetEncoderblock(in_channels, in_channels // 2, bn=self.bnmodel)
        the_channels = in_channels // 2

        self.nlsa512 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock512_0 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock512_1 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock512_2 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resjumpblock512d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.nlsa256 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock256_0 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock256_1 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock256_2 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resjumpblock256d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.nlsa128 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock128_0 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock128_1 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock128_2 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resjumpblock128d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.nlsa64 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock64_0 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock64_1 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock64_2 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)

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

        return out, r1024


class DRNLSA_i(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1, bn=0):
        super(DRNLSA_i, self).__init__()
        self.bnmodel = bn

        self.resjumpblock1024d = model_block.ResnetEncoderblock(in_channels, in_channels // 2, bn=self.bnmodel)
        the_channels = in_channels // 2

        self.nlsa512 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock512_0 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock512_1 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock512_2 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resjumpblock512d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.nlsa256 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock256_0 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock256_1 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resblock256_2 = model_block.Resnetlongblock(the_channels, bn=self.bnmodel)
        self.resjumpblock256d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.nlsa128 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock128_0 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock128_1 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock128_2 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resjumpblock128d = model_block.ResnetEncoderblock(the_channels, the_channels // 2, bn=self.bnmodel)
        the_channels = the_channels // 2

        self.nlsa64 = model_NLSA.NonLocalSparseAttention(the_channels)
        self.resblock64_0 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock64_1 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)
        self.resblock64_2 = model_block.Resnetshortblock(the_channels, bn=self.bnmodel)

        self.C0 = nn.Conv2d(the_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, inputs):
        rj1024d = self.resjumpblock1024d(inputs)

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


class DimensionReuctionNLSA_d(nn.Module):
    def __init__(self):
        super(DimensionReuctionNLSA_d, self).__init__()
        self.drnlsa = DRNLSA(1024, 1, 0)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inputsd = torch.squeeze(inputs, 1)

        drnlsaout, rec128 = self.drnlsa(inputsd)
        # restructureout = self.relu(drnlsaout)

        return drnlsaout, rec128


class DimensionReuctionNLSA_i(nn.Module):
    def __init__(self):
        super(DimensionReuctionNLSA_i, self).__init__()
        self.drnlsa = DRNLSA_i(1024, 1, 0)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):

        drnlsaout = self.drnlsa(inputs)

        return drnlsaout


class NLSA_Encoder(nn.Module):
    def __init__(self):
        super(NLSA_Encoder, self).__init__()
        self.drnlsa = DRNLSA(1024, 1, 0)
        self.drnlsa_i = DRNLSA_i(1024, 1, 0)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inputsd = torch.squeeze(inputs, 1)
        drnlsaout_d, rec1024 = self.drnlsa(inputsd)
        drnlsaout_i = self.drnlsa_i(rec1024)
        drnlsaout = torch.cat([drnlsaout_d, drnlsaout_i], dim=1)
        return drnlsaout, inputsd
