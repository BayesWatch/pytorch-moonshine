# blocks and convolution definitions
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

try:
    from pytorch_acdc.layers import FastStackedConvACDC
except ImportError:
    # then we assume you don't want to use this layer
    pass


def ACDC(in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=False):
    return FastStackedConvACDC(in_channels, out_channels, kernel_size, 12,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)

class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(Conv, self).__init__()
        # Dumb normal conv incorporated into a class
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class GConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, group_split, stride=1, kernel_size=3, padding=1, bias=False):
        super(GConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=bottleneck//group_split)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class AConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, groups, stride=1, kernel_size=3, padding=1, bias=False):
        super(AConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=groups)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class G2B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G2B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 2,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G4B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G4B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 4,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G8B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G8B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 8,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G16B2(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G16B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,group_split = 16,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)


class A2B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A2B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups = 2,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class A4B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A4B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups = 4,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class A8B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A8B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups= 8,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class A16B2(AConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(A16B2, self).__init__(in_planes, out_planes, bottleneck = out_planes // 2,groups = 16,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)


class G2B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G2B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 2,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G4B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G4B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 4,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G8B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G8B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 8,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)

class G16B4(GConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(G16B4, self).__init__(in_planes, out_planes, bottleneck = out_planes // 4,group_split = 16,
                                     stride=stride, kernel_size=kernel_size, padding=padding,
                                     bias=bias)



class ConvB2(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB2, self).__init__(in_planes, out_planes, out_planes//2,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB4(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB4, self).__init__(in_planes, out_planes, out_planes//4,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB8(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB8, self).__init__(in_planes, out_planes, out_planes//8,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class ConvB16(ConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(ConvB16, self).__init__(in_planes, out_planes, out_planes//16,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class Conv2x2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=2, padding=1, bias=False):
        super(Conv2x2, self).__init__()
        # Dilated 2x2 convs
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=2,
                              stride=stride, padding=padding, bias=bias, dilation=2)

    def forward(self, x):
        return self.conv(x)


class DConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False, groups=None):
        super(DConv, self).__init__()
        # This class replaces BasicConv, as such it assumes the output goes through a BN+ RELU whereas the
        # internal BN + RELU is written explicitly
        self.convdw = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=in_planes if groups is None else groups)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv1x1(F.relu(self.bn(self.convdw(x))))

class DConvG2(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG2, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//2)

class DConvG4(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG4, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//4)

class DConvG8(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG8, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//8)

class DConvG16(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvG16, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=in_planes//16)


class DConvA2(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA2, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=2)

class DConvA4(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA4, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=4)

class DConvA8(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA8, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=8)

class DConvA16(DConv):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvA16, self).__init__(in_planes, out_planes,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias, groups=16)


class DConvBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, bottleneck, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.convdw = nn.Conv2d(bottleneck, bottleneck, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias, groups=bottleneck)
        self.bn2= nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.convdw(out)))
        out = self.conv1x1_up(out)
        return out

class DConvB2(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB2, self).__init__(in_planes, out_planes, out_planes//2,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB4(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB4, self).__init__(in_planes, out_planes, out_planes//4,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB8(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB8, self).__init__(in_planes, out_planes, out_planes//8,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)

class DConvB16(DConvBottleneck):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConvB16, self).__init__(in_planes, out_planes, out_planes//16,
                stride=stride, kernel_size=kernel_size, padding=padding,
                bias=bias)


class DConv3D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False):
        super(DConv3D, self).__init__()
        # Separable conv approximating the 1x1 with a 3x3 conv3d
        self.convdw = nn.Conv2d(in_planes,in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias,groups=in_planes)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv3d = nn.Conv3d(1, out_planes, kernel_size=(3,1,1), stride=1, padding=(1,0,0), bias=bias)

    def forward(self, x):
        o = F.relu(self.bn(self.convdw(x)))
        o = o.unsqueeze(1)
        #n, c, d, w, h = o.size()
        return self.conv3d(o).mean(2)


def conv_function(convtype):
    if convtype == 'Conv':
        conv = Conv
    elif convtype == 'DConv':
        conv = DConv
    elif convtype == 'DConvG2':
        conv = DConvG2
    elif convtype == 'DConvG4':
        conv = DConvG4
    elif convtype == 'DConvG8':
        conv = DConvG8
    elif convtype == 'DConvG16':
        conv = DConvG16
    elif convtype == 'DConvA2':
        conv = DConvA2
    elif convtype == 'DConvA4':
        conv = DConvA4
    elif convtype == 'DConvA8':
        conv = DConvA8
    elif convtype == 'DConvA16':
        conv = DConvA16
    elif convtype == 'Conv2x2':
        conv = Conv2x2
    elif convtype == 'ConvB2':
        conv = ConvB2
    elif convtype == 'ConvB4':
        conv = ConvB4
    elif convtype == 'ConvB8':
        conv = ConvB8
    elif convtype == 'ConvB16':
        conv = ConvB16
    elif convtype == 'DConvB2':
        conv = DConvB2
    elif convtype == 'DConvB4':
        conv = DConvB4
    elif convtype == 'DConvB8':
        conv = DConvB8
    elif convtype == 'DConvB16':
        conv = DConvB16
    elif convtype == 'DConv3D':
        conv = DConv3D
    elif convtype =='G2B2':
        conv = G2B2
    elif convtype =='G4B2':
        conv = G4B2
    elif convtype =='G8B2':
        conv = G8B2
    elif convtype =='G16B2':
        conv = G16B2
    elif convtype =='G2B4':
        conv = G2B4
    elif convtype =='G4B4':
        conv = G4B4
    elif convtype =='G8B4':
        conv = G8B4
    elif convtype =='G16B4':
        conv = G16B4
    elif convtype =='A2B2':
        conv = A2B2
    elif convtype =='A4B2':
        conv = A4B2
    elif convtype =='A8B2':
        conv = A8B2
    elif convtype =='A16B2':
        conv = A16B2
    elif convtype =='ACDC':
        conv = ACDC
    else:
        raise ValueError('Conv "%s" not recognised'%convtype)
    return conv


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class OldBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv):

        super(OldBlock, self).__init__()

        self.conv1 = conv(in_planes, out_planes, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(out_planes, out_planes, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.equalInOut = (in_planes == out_planes)
        self.downsample = (not self.equalInOut or stride >1) and \
                          nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                padding=0, bias=False), nn.BatchNorm2d(out_planes)) \
                          or None
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv, xy=None):
        super(BottleBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = conv(out_planes, out_planes, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = out
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)




class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, conv = Conv):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, conv)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, conv):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, conv))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

