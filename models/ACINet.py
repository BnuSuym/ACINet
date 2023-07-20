# -*- coding: utf-8 -*-
"""
@Author: sym
@File: ACINet.py
@Time: 2023/5/15
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Extra_stage import FifthBlock
from MCAM import MCAM
from PEM import PEM
from P2T import p2t_small


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class ACINet(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm):
        super(ACINet, self).__init__()

        self.rgb_p2t = p2t_small()
        self.depth_p2t = p2t_small()

        # self.layer5 = layer(512, 8)
        self.layer5 = layer(512, 8)
        self.CIA_R1 = CIA_R(512, 512)
        self.CIA_R2 = CIA_R(512, 320)
        self.CIA_D1 = CIA_D(320, 128)
        self.CIA_D2 = CIA_D(128, 64)
        self.decoder = MCDD()

        self.conv128_1 = conv3x3(64, 1)
        self.conv256_1 = conv3x3(128, 1)
        self.conv512_1 = conv3x3(320, 1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.delayer_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.GELU(),
            self.upsample2
        )
        self.delayer_2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.delayer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2

        )

        self.prelayer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample4,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.conv320_1 = nn.Conv2d(320, 1, 3, padding=1)
        self.conv128_1 = nn.Conv2d(128, 1, 3, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x, d):
        rgb_list = self.rgb_p2t(x)
        depth_list = self.depth_p2t(d)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]
        r0 = self.layer5(r1)
        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]

        fuse1 = self.CIA_R1(r1, r0, d1)
        fuse2 = self.CIA_R2(r2, r1, d2)
        fuse3 = self.CIA_D1(r3, d3, d2)
        fuse4 = self.CIA_D2(r4, d4, d3)

        ou1, out2 = self.decoder(fuse1, fuse2, fuse3, fuse4)

        return ou1, out2

    def load_pre(self, pre_model):
        self.rgb_p2t.load_state_dict(torch.load(pre_model), strict=False)
        print(f"RGB p2t loading pre_model ${pre_model}")
        self.depth_p2t.load_state_dict(torch.load(pre_model), strict=False)
        print(f"Depth p2t loading pre_model ${pre_model}")


####################################################
class layer(nn.Module):
    def __init__(self, in_channel, num_head):
        super(layer, self).__init__()
        self.layer = FifthBlock(dim=in_channel, num_heads=num_head)

    def forward(self, x):
        x = self.layer(x)
        return x


class Conv1x1(nn.Module):
    def __init__(self, in_channel, out_channel, relu=True):
        super(Conv1x1, self).__init__()
        conv = [
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        ]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DWConv2(nn.Module):
    def __init__(self, in_channel, k=3, d=1, relu=False):
        super(DWConv2, self).__init__()
        conv = [
            nn.Conv2d(in_channel, in_channel, k, 1, (k // 2) * d, d, in_channel, False),
            nn.BatchNorm2d(in_channel)
        ]
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, d=1, relu_dw=False, relu_p=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            DWConv2(in_channel, 3, d, relu_dw),
            Conv1x1(in_channel, out_channel, relu_p)
        )

    def forward(self, x):
        return self.conv(x)


#####################################################
class CIA_D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CIA_D, self).__init__()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3 = conv3x3_bn_relu(in_channel + out_channel, out_channel)

        self.conv1x1_r = nn.Sequential(
            nn.Conv2d(out_channel, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv1x1_d = nn.Sequential(
            nn.Conv2d(out_channel, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.dep_ca = ChannelAttention(out_channel)
        self.rgb_ca = ChannelAttention(out_channel)
        self.conv_cat = DSConv3x3(out_channel * 2, out_channel)
        # self.block = Block(in_channel)

    def forward(self, r, d1, d2):
        d2 = self.upsample2(d2)
        d = torch.cat((d1, d2), dim=1)
        d = self.conv3x3(d)
        assert r.shape == d.shape, "rgb and depth should have same size"

        r_s_en = r + r * self.conv1x1_r(d)
        d_s_en = d + d * self.conv1x1_d(r)

        r_c_en = r * self.rgb_ca(r_s_en)
        d_c_en = d * self.dep_ca(d_s_en)

        rd_en = r_c_en * d_c_en
        rgb_en = r_c_en + rd_en
        dep_en = d_c_en + rd_en
        out = self.conv_cat(torch.cat((rgb_en, dep_en), 1))
        # out = self.block(out)
        return out


class CIA_R(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CIA_R, self).__init__()
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3_1 = conv3x3_bn_relu(in_channel + out_channel, out_channel)
        self.conv3x3_2 = conv3x3_bn_relu(in_channel + out_channel, out_channel)
        self.conv1x1_r = nn.Sequential(
            nn.Conv2d(out_channel, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv1x1_d = nn.Sequential(
            nn.Conv2d(out_channel, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.dep_ca = ChannelAttention(out_channel)
        self.rgb_ca = ChannelAttention(out_channel)
        self.conv_cat = DSConv3x3(out_channel * 2, out_channel)
        # self.block = Block(in_channel)

    def forward(self, r1, r2, d):
        if r1.shape == r2.shape:
            r = torch.cat((r1, r2), dim=1)
            r = self.conv3x3_1(r)
        else:
            r2 = self.upsample2(r2)
            r = torch.cat((r1, r2), dim=1)
            r = self.conv3x3_2(r)
        assert r.shape == d.shape, "rgb and depth should have same size"

        r_s_en = r + r * self.conv1x1_r(d)
        d_s_en = d + d * self.conv1x1_d(r)

        r_c_en = r * self.rgb_ca(r_s_en)
        d_c_en = d * self.dep_ca(d_s_en)

        rd_en = r_c_en * d_c_en
        rgb_en = r_c_en + rd_en
        dep_en = d_c_en + rd_en
        out = self.conv_cat(torch.cat((rgb_en, dep_en), 1))
        # out = self.block(out)
        return out


######################################################################################
class MCDD(nn.Module):
    def __init__(self):
        super(MCDD, self).__init__()

        self.mcam1 = MCAM(512, 512, 320)
        self.mcam2 = MCAM(512, 320, 128)
        self.mcam3 = MCAM(320, 128, 64)
        self.mcam4 = MCAM(128, 64, 64)
        self.pem1 = PEM(512)
        self.pem2 = PEM(320)
        self.pem3 = PEM(128)
        self.pem4 = PEM(64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.delayer_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.GELU(),
            self.upsample2
        )
        self.delayer_2 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.delayer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.delayer_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=320, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.GELU(),
            self.upsample2
        )
        self.delayer_5 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.delayer_6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )

        self.prelayer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample4,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        self.prelayer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample4,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4):  # x1 x2 x3 x4 为融合特征 从高级到低级
        fuse1 = self.mcam1(in1=x1, in3=x2)
        fuse2 = self.mcam2(in1=x2, in2=x1, in3=x3)
        fuse3 = self.mcam3(in1=x3, in2=x2, in3=x4)
        fuse4 = self.mcam4(in1=x4, in2=x3)

        de1_1 = fuse1
        de1_2 = self.delayer_1(de1_1) + fuse2
        de1_3 = self.delayer_2(de1_2) + fuse3
        de1_4 = self.delayer_3(de1_3) + fuse4
        sal_out = self.prelayer1(de1_4)

        sal_att1 = F.interpolate(sal_out, size=(12, 12), mode='bilinear', align_corners=True)
        sal_att2 = F.interpolate(sal_out, size=(24, 24), mode='bilinear', align_corners=True)
        sal_att3 = F.interpolate(sal_out, size=(48, 48), mode='bilinear', align_corners=True)
        sal_att4 = F.interpolate(sal_out, size=(96, 96), mode='bilinear', align_corners=True)

        de2_1 = self.delayer_4(self.pem1(fuse1, sal_att1))
        de2_2 = self.delayer_5(self.pem2(de1_2, sal_att2) + de2_1)
        de2_3 = self.delayer_6(self.pem3(de1_3, sal_att3) + de2_2)
        sal_out2 = self.prelayer2(self.pem4(de1_4, sal_att4) + de2_3)

        return sal_out, sal_out2

#####################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    pre_path = './pretrain/p2t_small.pth'
    a = np.random.random((1, 3, 224, 224))
    b = np.random.random((1, 3, 224, 224))
    c = torch.Tensor(a).cuda()
    d = torch.Tensor(b).cuda()
    ACINet = ACINet().cuda()
    ACINet.load_pre(pre_path)
    ACINet(c, d)
