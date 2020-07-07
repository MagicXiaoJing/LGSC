import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from model.resnet import  ResNet

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_func = nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_func(planes, affine=True)
        self.relu = nn.PReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_func(planes, affine=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Conv_Block(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, norm_func=None, use_act=True):
        super(Conv_Block, self).__init__()
        process_list = []
        if kernel_size==3:
            process_list.append(conv3x3(inplanes, planes))
        elif kernel_size == 1:
            process_list.append(conv1x1(inplanes, planes))
        if norm_func is not None:
            process_list.append(norm_func(planes, affine=True))
        if use_act:
            process_list.append(nn.PReLU())
        self.process_list = nn.Sequential(*process_list)

    def forward(self, x):
        x = self.process_list(x)
        return x

class Upsample(nn.Module):
    def __init__(self, inplanes, concat_planes, out_planes, bilinear=False):
        super(Upsample, self).__init__()
        self.bilinear = nn.UpsamplingNearest2d(scale_factor=2)
        self.norm = nn.InstanceNorm2d # nn.BatchNorm2d#
        if inplanes>64:
            self.conv_half = Conv_Block(inplanes, int(inplanes/2), norm_func=self.norm)
            self.conv2 = Conv_Block(int(inplanes / 2) + concat_planes, out_planes, kernel_size=1, norm_func=self.norm, use_act=False)
            self.block = self._make_layer(BasicBlock, int(inplanes / 2) + concat_planes, out_planes, 2)
        else:
            self.conv_half = Conv_Block(inplanes, inplanes, norm_func=self.norm)
            self.conv2 = Conv_Block(inplanes + concat_planes, out_planes, kernel_size=1, norm_func=self.norm, use_act=False)
            self.block = self._make_layer(BasicBlock, inplanes + concat_planes, out_planes, 2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                self.norm(planes * block.expansion, affine=True),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_func=self.norm))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_func=self.norm))

        return nn.Sequential(*layers)

    def forward(self, x1, x2=None):
        if x2 is not None:
            x1 = self.bilinear(x1)
            x1 = self.conv_half(x1)
            x = torch.cat([x1, x2], dim=1)
            out = self.block(x)+self.conv2(x)
        else:
            x1 =  self.bilinear(x1)
            x1 = self.conv_half(x1)
            identity = self.conv2(x1)
            out = self.block(x1) + identity

        return out



class U_Net(nn.Module):

    def __init__(self):
        super(U_Net, self).__init__()
        self.backbone = ResNet("resnet18", 1)
        self.tanh = nn.Tanh()
        self.upsample_func = Upsample


        self.Upsample1 = self.upsample_func(512, 256, 256)
        self.Upsample2 = self.upsample_func(256, 128, 128)
        self.Upsample3 = self.upsample_func(128, 64, 64)
        self.Upsample4 = self.upsample_func(64, 64, 64)
        self.Upsample5 = self.upsample_func(64, 0, 3)

    def forward(self, x):
        x, Down1, Down2, Down3, Down4 = self.backbone(x)
        Up1 = self.Upsample1(Down4, Down3)
        Up2 = self.Upsample2(Up1, Down2)
        Up3 = self.Upsample3(Up2, Down1)
        Up4 = self.Upsample4(Up3, x)
        Up5 = self.Upsample5(Up4)
        out = self.tanh(Up5)
        if self.training:
            return Down4, Up1, Up2, Up3, Up4, out
        else:
            return out

