import torch
import torch.nn as nn
import torch.nn.init as init
from models.batchrenorm import BatchRenorm2d


class MsFeat(nn.Module):
    def __init__(self, in_channels):
        outchannel_MS = 2
        super(MsFeat, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1, 1, 1), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv1[0].bias, 0.0)

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, outchannel_MS, 3, stride=(1, 1, 1), padding=2, dilation=2, bias=True),
            nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv2[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv2[0].bias, 0.0)

        self.conv3 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=1, dilation=1, bias=True),
                                   nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv3[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv3[0].bias, 0.0)

        self.conv4 = nn.Sequential(nn.Conv3d(outchannel_MS, outchannel_MS, 3, padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv4[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv4[0].bias, 0.0)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(inputs)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv1)
        return torch.cat((conv1, conv2, conv3, conv4), 1)


class NonLocal(nn.Module):
    def __init__(self, inplanes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(NonLocal, self).__init__()
        # conv theta
        self.t = nn.Conv3d(inplanes, inplanes // 1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.t.weight, 0, 'fan_in', 'relu')
        # conv phi
        self.p = nn.Conv3d(inplanes, inplanes // 1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.p.weight, 0, 'fan_in', 'relu')
        # conv g
        self.g = nn.Conv3d(inplanes, inplanes // 1, kernel_size=1, stride=1, bias=False)
        init.kaiming_normal_(self.g.weight, 0, 'fan_in', 'relu')
        # conv z
        self.z = nn.Conv3d(inplanes // 1, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        init.kaiming_normal_(self.z.weight, 0, 'fan_in', 'relu')
        # concat groups
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)
        init.constant_(self.gn.weight, 0)
        nn.init.constant_(self.gn.bias, 0)

        if self.use_scale:
            print("=> WARN: Non-local block uses 'SCALE'")
        if self.groups:
            print("=> WARN: Non-local block uses '{}' groups".format(self.groups))

    def kernel(self, t, p, g, b, c, d, h, w):
        """The linear kernel (dot production).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            d: depth of featuremaps
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * d * h * w)
        p = p.view(b, 1, c * d * h * w)
        g = g.view(b, c * d * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c * d * h * w) ** 0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, d, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)  # b,ch,d,h,w
        p = self.p(x)  # b,ch,d,h,w
        g = self.g(x)  # b,ch,d,h,w

        b, c, d, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, d, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, d, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


# feature integration
class Block(nn.Module):
    def __init__(self, in_channels):
        outchannel_block = 16
        super(Block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, outchannel_block, 1, padding=0, dilation=1, bias=True),
                                   nn.ReLU(inplace=True))
        init.kaiming_normal_(self.conv1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.conv1[0].bias, 0.0)

        self.feat1 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=1, dilation=1, bias=True),
                                   nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat1[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.feat1[0].bias, 0.0)

        self.feat15 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=2, dilation=2, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat15[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.feat15[0].bias, 0.0)

        self.feat2 = nn.Sequential(nn.Conv3d(outchannel_block, 8, 3, padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat2[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.feat2[0].bias, 0.0)

        self.feat25 = nn.Sequential(nn.Conv3d(8, 4, 3, padding=1, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat25[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.feat25[0].bias, 0.0)

        self.feat = nn.Sequential(nn.Conv3d(24, 8, 1, padding=0, dilation=1, bias=True), nn.ReLU(inplace=True))
        init.kaiming_normal_(self.feat[0].weight, 0, 'fan_in', 'relu')
        init.constant_(self.feat[0].bias, 0.0)

    # note the channel for each layer
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        feat1 = self.feat1(conv1)
        feat15 = self.feat15(feat1)
        feat2 = self.feat2(conv1)
        feat25 = self.feat25(feat2)
        feat = self.feat(torch.cat((feat1, feat15, feat2, feat25), 1))
        return torch.cat((inputs, feat), 1)


class Resnetshortblock(nn.Module):
    def __init__(self, in_channels=64, bn=0):
        super(Resnetshortblock, self).__init__()
        if bn == 0:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1))
        elif bn == 1:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels))
        else:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                BatchRenorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                BatchRenorm2d(in_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, flag_res=1):
        resblock_out = self.resblock(inputs)
        if flag_res == 0:
            output = self.relu(resblock_out)
            return output
        out = resblock_out + inputs
        output = self.relu(out)
        return output


class Resnetlongblock(nn.Module):
    def __init__(self, in_channels, bn=0):
        super(Resnetlongblock, self).__init__()
        if (in_channels // 4) < 64:
            temp_channels = 64
        else:
            temp_channels = in_channels // 4
        if bn == 0:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, temp_channels, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, temp_channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, in_channels, 1, 1, 0, bias=False))
        elif bn == 1:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, temp_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, temp_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, in_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(in_channels))
        else:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, temp_channels, 1, 1, 0, bias=False),
                BatchRenorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, temp_channels, 3, 1, 1, bias=False),
                BatchRenorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, in_channels, 1, 1, 0, bias=False),
                BatchRenorm2d(in_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, flag_res=1):
        resblock_out = self.resblock(inputs)
        if flag_res == 0:
            output = self.relu(resblock_out)
            return output
        out = resblock_out + inputs
        output = self.relu(out)
        return output


class ResnetEncoderblock(nn.Module):
    def __init__(self, in_channels, out_channels, bn=0):
        super(ResnetEncoderblock, self).__init__()
        if (in_channels // 4) < 64:
            temp_channels = 64
        else:
            temp_channels = in_channels // 4
        if bn == 0:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, temp_channels, 1, 1, 0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, temp_channels, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, out_channels, 1, 1, 0, bias=False))
            self.jumpblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))
        elif bn == 1:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, temp_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, temp_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels))
            self.jumpblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.resblock = nn.Sequential(
                nn.Conv2d(in_channels, temp_channels, 1, 1, 0, bias=False),
                BatchRenorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, temp_channels, 3, 1, 1, bias=False),
                BatchRenorm2d(temp_channels), nn.ReLU(inplace=True),
                nn.Conv2d(temp_channels, out_channels, 1, 1, 0, bias=False),
                BatchRenorm2d(out_channels))
            self.jumpblock = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                BatchRenorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, flag_res=1):
        resblock_out = self.resblock(inputs)
        if flag_res == 0:
            output = self.relu(resblock_out)
            return output
        jump_out = self.jumpblock(inputs)
        out = resblock_out + jump_out
        output = self.relu(out)
        return output


class DownD(nn.Module):
    def __init__(self, in_channels=1024, last_channels=64, out_channels=1):
        super(DownD, self).__init__()
        layers = []
        while in_channels > last_channels:
            layers.append(nn.Conv2d(in_channels, int(in_channels / 2), 3, 1, 1, bias=False))
            layers.append(BatchRenorm2d(int(in_channels / 2)))
            layers.append(nn.ReLU(inplace=True))
            in_channels = int(in_channels / 2)
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
        self.downd = nn.Sequential(*layers)

    def forward(self, inputs):
        downdimension_out = self.downd(inputs)
        return downdimension_out


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, bias=True,
            bn=False, act=nn.PReLU()):

        m = [nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
