from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch



__all__ = ['ResNetBB', 'resnet18bb', 'resnet34bb', 'resnet50bb', 'resnet101bb',
           'resnet152bb']


class ResNetBB(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, num_features=0, norm=False, dropout=0, **kwargs):
        super(ResNetBB, self).__init__()

        self.pretrained = pretrained
        self.depth = depth
        # Construct base (pretrained) resnet
        if depth not in ResNetBB.__factory: raise KeyError("Unsupported depth:", depth)
        if depth >= 50: resnet = ResNetBB.__factory[depth](pretrained="IMAGENET1K_V2") #(pretrained=pretrained)
        else: resnet = ResNetBB.__factory[depth](pretrained=pretrained)
        if depth >= 50:
            resnet.layer4[0].conv2.stride = (1,1)
            resnet.layer4[0].downsample[0].stride = (1,1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveMaxPool2d(1) #nn.AdaptiveAvgPool2d

        
        self.num_features = num_features
        self.norm = norm
        self.has_embedding = num_features > 0
        self.out_planes = out_planes = resnet.fc.in_features

        # Append new Last FC layers
        if self.has_embedding:
            self.feat = nn.Linear(out_planes, self.num_features)
            self.feat_bn = nn.BatchNorm1d(self.num_features)
            init.kaiming_normal_(self.feat.weight, mode='fan_out')
            init.constant_(self.feat.bias, 0)
        else:
            # Change the num_features to CNN output channels
            self.num_features = out_planes
            self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False, joint=False):
        x = self.base(x) # [bs, 2048, 16, 8]
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.has_embedding: bn_x = self.feat_bn(self.feat(x))
        else: bn_x = self.feat_bn(x)

        if self.training is False:
            bn_x = F.normalize(bn_x)
            return bn_x

        return x, bn_x



    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNetBB.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.relu.state_dict())
        self.base[3].load_state_dict(resnet.maxpool.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())

def resnet18bb(**kwargs):
    return ResNetBB(18, **kwargs)


def resnet34bb(**kwargs):
    return ResNetBB(34, **kwargs)


def resnet50bb(**kwargs):
    return ResNetBB(50, **kwargs)


def resnet101bb(**kwargs):
    return ResNetBB(101, **kwargs)


def resnet152bb(**kwargs):
    return ResNetBB(152, **kwargs)
