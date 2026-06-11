import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNNProjectionLayer(nn.Module):
    def __init__(self, in_channels, block=BasicBlock, num_blocks=None, avg_pool=True, flatten=True):
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layers = nn.Sequential()
        for i, nb in enumerate(num_blocks):
            stride = 1 if i == 0 else 2
            self.layers.add_module(f"layer{i+1}", self._make_layer(block, 64 * (2**i), nb, stride))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else nn.Identity()
        self.linear_proj = nn.Identity()
        self.proj_norm = nn.Identity()
        self.flatten = flatten

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layers(out)
        out = self.avgpool(out)
        if self.flatten:
            out = out.view(out.size(0), -1)
        out = self.linear_proj(out)
        out = self.proj_norm(out)
        return out


def get_projection_model(config, num_input_channels):
    num_blocks = config["cnn_proj_resnet_num_blocks"]
    proj_type = config["proj_type"]

    if proj_type == "cnn_proj_resnet_with_avg_pool":
        return CNNProjectionLayer(num_input_channels, BasicBlock, num_blocks, avg_pool=True, flatten=True)

    elif proj_type == "cnn_proj_resnet_with_avg_pool_linear_proj":
        model = CNNProjectionLayer(num_input_channels, BasicBlock, num_blocks, avg_pool=True, flatten=True)
        model.linear_proj = nn.Linear(512, config["proj_dims"], bias=False)
        return model

    elif proj_type == "resnet18_wth_IMAGENET_weights":
        model = resnet18(weights="DEFAULT")
        model.conv1 = nn.Conv2d(num_input_channels, 64, 7, stride=2, padding=3, bias=False)
        model.fc = nn.Identity()
        return model

    elif proj_type == "cnn_proj_resnet":
        return CNNProjectionLayer(num_input_channels, BasicBlock, num_blocks, avg_pool=False, flatten=False)

    elif proj_type == "cnn_proj_resnet_linear_proj":
        model = CNNProjectionLayer(num_input_channels, BasicBlock, num_blocks, avg_pool=False, flatten=False)
        out_ch = 2 ** (len(num_blocks) - 1) * 64
        model.linear_proj = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        return model

    elif proj_type == "cnn_proj_resnet_linear_proj_batch_norm":
        model = CNNProjectionLayer(num_input_channels, BasicBlock, num_blocks, avg_pool=False, flatten=False)
        out_ch = 2 ** (len(num_blocks) - 1) * 64
        model.linear_proj = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        model.proj_norm = nn.BatchNorm2d(out_ch, affine=False)
        return model

    elif proj_type == "cnn_proj_resnet_batch_norm":
        model = CNNProjectionLayer(num_input_channels, BasicBlock, num_blocks, avg_pool=False, flatten=False)
        out_ch = 2 ** (len(num_blocks) - 1) * 64
        model.proj_norm = nn.BatchNorm2d(out_ch, affine=False)
        return model

    else:
        raise ValueError(f"Unknown proj_type: {proj_type}")
