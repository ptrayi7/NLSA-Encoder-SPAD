import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from models import model_block


# build the model
class DeepBoosting(nn.Module):
    def __init__(self, in_channels=1):
        super(DeepBoosting, self).__init__()
        # 进入特征提取
        self.msfeat = model_block.MsFeat(in_channels)
        self.C0 = nn.Sequential(nn.Conv3d(8, 2, kernel_size=1, stride=(1, 1, 1), bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.C0[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.C0[0].bias, 0.0)
        self.nl = model_block.NonLocal(2, use_scale=False, groups=1)

        self.convrup = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1), bias=True))
        init.kaiming_normal_(self.convrup[0].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convrup[2].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convrup[4].weight, 0, 'fan_in', 'relu')
        init.normal_(self.convrup[6].weight, mean=0.0, std=0.001)

        self.dfus_block0 = model_block.Block(32)
        self.dfus_block1 = model_block.Block(40)
        self.dfus_block2 = model_block.Block(48)
        self.dfus_block3 = model_block.Block(56)
        self.dfus_block4 = model_block.Block(64)
        self.dfus_block5 = model_block.Block(72)
        self.dfus_block6 = model_block.Block(80)
        self.dfus_block7 = model_block.Block(88)
        self.dfus_block8 = model_block.Block(96)
        self.dfus_block9 = model_block.Block(104)
        self.convrdown = nn.Sequential(
            nn.ConvTranspose3d(112, 56, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(56, 28, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(28, 14, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(14, 7, kernel_size=(6, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1), bias=False))
        init.kaiming_normal_(self.convrdown[0].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convrdown[2].weight, 0, 'fan_in', 'relu')
        init.kaiming_normal_(self.convrdown[4].weight, 0, 'fan_in', 'relu')
        init.normal_(self.convrdown[6].weight, mean=0.0, std=0.001)

        self.C1 = nn.Sequential(nn.Conv3d(7, 1, kernel_size=1, stride=(1, 1, 1), bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.C1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.C1[0].bias, 0.0)

    def forward(self, inputs):
        smax = torch.nn.Softmax2d()
        msfeat = self.msfeat(inputs)
        c0 = self.C0(msfeat)
        nlout = self.nl(c0)
        convrupout = self.convrup(nlout)
        b0 = self.dfus_block0(convrupout)
        b1 = self.dfus_block1(b0)
        b2 = self.dfus_block2(b1)
        b3 = self.dfus_block3(b2)
        b4 = self.dfus_block4(b3)
        b5 = self.dfus_block5(b4)
        b6 = self.dfus_block6(b5)
        b7 = self.dfus_block7(b6)
        b8 = self.dfus_block8(b7)
        b9 = self.dfus_block9(b8)
        convrdownout = self.convrdown(b9)
        convrout = self.C1(convrdownout)

        denoise_out = torch.squeeze(convrout, 1)

        weights = Variable(
            torch.linspace(0, 1, steps=denoise_out.size()[1]).unsqueeze(1).unsqueeze(1).type(torch.cuda.FloatTensor))
        weighted_smax = weights * smax(denoise_out)
        soft_argmax = weighted_smax.sum(1).unsqueeze(1)

        return soft_argmax, denoise_out
