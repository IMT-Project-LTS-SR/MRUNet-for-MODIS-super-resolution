import torch
import cv2
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import os
import torch.nn.functional as F

# VDSR and DMCN model

from math import sqrt

import torch.nn.init as init

num = 64

class DwSample(nn.Module):
    def __init__(self, inp, oup, stride, kernal_size = 3, groups=1, BN = False):
        super(DwSample, self).__init__()
        if BN == True:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
                nn.BatchNorm2d(oup),
                nn.PReLU(),
            )
        else:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size-1)/2), groups=groups),
                nn.PReLU(),
            )

    def forward(self, x):
        residual = x
        out = self.conv_dw(x)
        return torch.add(out, residual)

class BasicBlock(nn.Module):
    def __init__(self, inp, oup, stride, kernal_size=3, groups=1, BN = False):
        super(BasicBlock, self).__init__()
        if BN == True:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
                nn.BatchNorm2d(oup),
                nn.PReLU(),
                nn.Conv2d(oup, inp, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
            )
        else:
            self.conv_dw = nn.Sequential(
                nn.Conv2d(inp, oup, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
                nn.PReLU(),
                nn.Conv2d(oup, inp, kernal_size, stride, int((kernal_size - 1) / 2), groups=groups),
            )
    def forward(self, x):
        residual = x
        return torch.add(self.conv_dw(x), residual)

class UpSample(nn.Module):
    def __init__(self, f, upscale_factor):
        super(UpSample, self).__init__()

        self.relu = nn.PReLU()
        self.conv = nn.Conv2d(f, f * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pixel_shuffle(x)
        return x

class DMCN_prelu(nn.Module):
    def __init__(self, BN=True, width = 64):
        super(DMCN_prelu, self).__init__()
        self.input1 = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.input2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(width)
        self.input3 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(width)
        self.input4 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN3 = nn.BatchNorm2d(width)
        self.input5 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN4 = nn.BatchNorm2d(width)
        self.down_sample1 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1, bias=False)
        self.Conv_DW_layers1 = self.make_layer(DwSample, 5, BN, width)

        self.down_sample2 = nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1, bias=False)
        self.Conv_DW_layers2 = self.make_layer(DwSample, 2, BN, width)

        self.up_sample1 = UpSample(width,2)

        self.choose1 = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1, stride=1, padding=0, bias=False)
        self.resudial_layers1 = self.make_layer(BasicBlock, 2, BN, width)

        self.up_sample2 = UpSample(width,2)

        self.choose2 = nn.Conv2d(in_channels=width*2, out_channels=width, kernel_size=1, stride=1, padding=0, bias=False)
        self.resudial_layers2 = self.make_layer(BasicBlock, 5, BN, width)

        self.output = nn.Conv2d(in_channels=width, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.PReLU()

    def make_layer(self, block, num_of_layer, BN, width):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(width, width, 1, 3, 1, BN))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        s1 = self.relu(self.input1(x))
        s1 = self.input2(s1)
        s1 = self.relu(self.BN1(s1))
        s1 = self.input3(s1)
        s1 = self.relu(self.BN2(s1))
        s1 = self.input4(s1)
        s1 = self.relu(self.BN3(s1))
        s1 = self.input5(s1)
        s1 = self.relu(self.BN4(s1))
        out = self.down_sample1(s1)
        s2 = self.Conv_DW_layers1(out)

        out = self.down_sample2(s2)
        out = self.Conv_DW_layers2(out)

        out = self.up_sample1(out)
        out = torch.cat((s2, out), 1)
        out = self.choose1(out)
        out = self.resudial_layers1(out)

        out = self.up_sample2(out)
        out = torch.cat((s1, out), 1)
        out = self.choose2(out)
        out = self.resudial_layers2(out)

        out = self.output(out)
        out = torch.add(out, residual)
        return out


""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_Down(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv_Down, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, res_down=False):
        super(Down, self).__init__()
        self.res_down = res_down
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        if self.res_down:
            self.in_conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.mid_conv = DoubleConv(in_channels, in_channels)
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.res_down:
            return self.out_conv(self.mid_conv((self.in_conv(x))) + self.in_conv(x))

        else:
            return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2],mode='reflect')
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #     nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, res_down=False, n_resblocks=1, padding_type="reflect", norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=True, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        ### Encoder
        self.down1 = Down(64, 128, res_down=res_down)
        self.down2 = Down(128, 256, res_down=res_down)
        self.down3 = Down(256, 512, res_down=res_down)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, res_down=res_down)

        ### Residual blocks
        resblocks = []
        for i in range(n_resblocks):
            resblocks += [ResnetBlock(1024 // factor, padding_type, norm_layer, use_dropout, use_bias)]
        self.resblocks = nn.Sequential(*resblocks)

        ### Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_dc = DoubleConv(64, 64)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x1 = self.inc(x) #64
        x2 = self.down1(x1) #128
        x3 = self.down2(x2) #256 
        x4 = self.down3(x3) #512
        x5 = self.down4(x4) #1024

        x5 = self.resblocks(x5)

        xp1 = self.up1(x5, x4) #512
        xp2 = self.up2(xp1, x3) #256
        xp3 = self.up3(xp2, x2) #128
        xp4 = self.up4(xp3, x1) #64
        # x = self.up(x)
        # x = self.up_dc(x)
        logits = self.outc(xp4)+x
        return logits