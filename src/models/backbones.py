import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class BackboneMLPBN(nn.Module):
    def __init__(self, dims, num_classes, normalize=False):
        super().__init__()
        self.normalize = normalize
        dims = list(dims) + [num_classes]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return self.layers(x)


class GroupBatchNorm2d(nn.BatchNorm2d):
    """BN that normalizes per-modality group within a concatenated batch."""
    def __init__(self, num_features, batch_size, **kwargs):
        super().__init__(num_features, **kwargs)
        self.batch_size = batch_size

    def forward(self, x):
        self._check_input_dim(x)
        ema = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                ema = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum

        t = x.view(-1, self.batch_size, self.num_features, x.shape[-2], x.shape[-1])
        if self.training:
            mean = t.mean([1, 3, 4])
            var = t.var([1, 3, 4], unbiased=False)
            n = x.numel() / x.size(1)
            with torch.no_grad():
                self.running_mean = ema * mean.mean(0) + (1 - ema) * self.running_mean
                self.running_var = ema * var.mean(0) * n / (n - 1) + (1 - ema) * self.running_var
        else:
            mean = self.running_mean[None, :]
            var = self.running_var[None, :]

        t = (t - mean[:, None, :, None, None]) / (torch.sqrt(var[:, None, :, None, None] + self.eps))
        if self.affine:
            x = t.view_as(x) * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return x


class BasicBlockGroupBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_size=None):
        super().__init__()
        if batch_size is None:
            raise ValueError("batch_size required for GroupBatchNorm2d")
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.gbn1 = GroupBatchNorm2d(planes, batch_size)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.gbn2 = GroupBatchNorm2d(planes, batch_size)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                GroupBatchNorm2d(planes, batch_size),
            )

    def forward(self, x):
        out = F.relu(self.gbn1(self.conv1(x)))
        out = self.gbn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_size=None):
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


class CNNBackbone(nn.Module):
    def __init__(self, in_channels, block, num_blocks, num_classes, batch_size=None, normalize=False):
        super().__init__()
        self.in_planes = in_channels
        self.batch_size = batch_size
        self.normalize = normalize
        self.layers = nn.Sequential()
        for i, nb in enumerate(num_blocks):
            stride = 1 if i == 0 else 2
            out_ch = in_channels * (2 ** (i + 1))
            self.layers.add_module(f"layer{i+1}", self._make_layer(block, out_ch, nb, stride))
        self.linear = nn.Linear(in_channels * (2 ** len(num_blocks)), num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s, self.batch_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        out = self.layers(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


def replace_gbn_with_bn(module):
    """Replace GroupBatchNorm2d (and reset BN running stats) for per-modality BN calibration."""
    for name, child in module.named_children():
        if isinstance(child, (GroupBatchNorm2d, nn.BatchNorm2d)):
            new_bn = nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=None,  # cumulative moving average
                affine=child.affine,
                track_running_stats=child.track_running_stats,
            )
            if child.affine:
                new_bn.weight.data.copy_(child.weight.data)
                new_bn.bias.data.copy_(child.bias.data)
            setattr(module, name, new_bn)
        else:
            replace_gbn_with_bn(child)


def recompute_batchnorm_running_stats(backbone, proj, dataloader, device, cal_epochs=10):
    """Post-hoc BN calibration: frozen weights, recompute per-modality BN stats."""
    backbone.train()
    proj.train()
    backbone.to(device)
    proj.to(device)
    with torch.no_grad():
        for _ in tqdm(range(cal_epochs), desc="BN calibration"):
            for data, _ in dataloader:
                data = data.to(device)
                _ = backbone(proj(data))


def get_backbone_model(config, num_classes):
    if config["backbone"] == "MLPBN":
        return BackboneMLPBN(dims=list(config["backbone_dims"]), num_classes=num_classes)
    elif config["backbone"] == "MLPBN_with_emb_norm":
        return BackboneMLPBN(dims=list(config["backbone_dims"]), num_classes=num_classes, normalize=True)
    elif config["backbone"] == "cnn_backbone_resnet_groupbn":
        in_ch = 2 ** (len(config["cnn_proj_resnet_num_blocks"]) - 1) * 64
        return CNNBackbone(in_ch, BasicBlockGroupBN, config["cnn_backbone_resnet_num_blocks"],
                           num_classes, batch_size=config["batch_size"],
                           normalize=config.get("normalize_backbone", False))
    elif config["backbone"] == "cnn_backbone_resnet":
        in_ch = 2 ** (len(config["cnn_proj_resnet_num_blocks"]) - 1) * 64
        return CNNBackbone(in_ch, BasicBlock, config["cnn_backbone_resnet_num_blocks"],
                           num_classes, batch_size=config.get("batch_size"),
                           normalize=config.get("normalize_backbone", False))
    else:
        raise ValueError(f"Unknown backbone: {config['backbone']}")
