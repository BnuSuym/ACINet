"""
@Author: sym
@File: CAU.py
@Time: 2024/7/15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
 

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


class CAU(nn.Module):
    def __init__(self, inchannel1, inchannel2, inchannel3, ratio=8):  # 输入通道数依次为低，中，高
        super(CAU, self).__init__()
        self.conv_cross = conv3x3_bn_relu(inchannel2 * 3, inchannel2)
        self.conv_up = conv3x3_bn_relu(inchannel1, inchannel2)
        self.conv_down = conv3x3_bn_relu(inchannel3, inchannel2)
        self.gam = GCA(inchannel2)

        self.eps = 1e-5

    def forward(self, in1, in2=None, in3=None, flag=None):  # 输入特征按照，中、高、低的顺序
        if in2 is not None and in1.size()[2:] != in2.size()[2:]:
            in2 = self.conv_up(in2)
            in2 = F.interpolate(in2, size=in1.size()[2:], mode='bilinear')
        else:
            in2 = in1
        if in3 is not None and in1.size()[2:] != in3.size()[2:]:
            in3 = self.conv_down(in3)
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
        else:
            in3 = in1

        x = torch.cat((in1, in2, in3), 1)
        x = self.conv_cross(x)  # [B, C, H, W]
        out = self.gam(x)

        return out


class GCA(nn.Module):
    """Global channel attention module"""

    def __init__(self, in_dim):
        super(GCA, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
