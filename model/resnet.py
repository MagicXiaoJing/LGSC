import torch.nn as nn
import torch
import math
import time
import torch.utils.model_zoo as model_zoo
from util.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
import os


class ResNet(nn.Module):
    # dict key: block, layer, pretrain_url
    arch_settings = {
        'resnet18': (BasicBlock, (2, 2, 2, 2), 'https://download.pytorch.org/models/resnet18-5c106cde.pth'),
        'resnet34': (BasicBlock, (3, 4, 6, 3), 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'),
        'resnet50': (Bottleneck, (3, 4, 6, 3), 'https://download.pytorch.org/models/resnet50-19c8e357.pth'),
        'resnet101': (Bottleneck, (3, 4, 23, 3), 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'),
        'resnet152': (Bottleneck, (3, 8, 36, 3), 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'),
    }

    # name
    def __init__(self, backbone_feature='resnet101', frozen_stages=-1, use_pretrain=True, use_all_indice=True):
        if backbone_feature in ResNet.arch_settings:
            dict_val = ResNet.arch_settings[backbone_feature]
            block = dict_val[0]
            layers = dict_val[1]
            url = dict_val[2]
            pretrain_model_dir = "/home/yanghf/Documents/myDisk/21Tesla1/Project/pretrain_model"
            if not os.path.exists(pretrain_model_dir):
                pretrain_model_dir = "/data1/yhf/pyTorch/Project/pretrain_model"

        else:
            raise ValueError(" not support this resnet arch, "
                             "write wrong???")
        self.use_all_indice = use_all_indice
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.frozen_stages = frozen_stages
        self.freeze_bn = False
        self._freeze_stages()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # load pretrained model
        if use_pretrain:
            print('use prepratin model')
            self.load_state_dict(model_zoo.load_url(url, model_dir=pretrain_model_dir), strict=False)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            for param in m.parameters():
                param.requires_grad = False

        self.freeze_bn = True

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        img_batch = inputs
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if self.use_all_indice:
            return x0, x1, x2, x3, x4
        else:
            return x4