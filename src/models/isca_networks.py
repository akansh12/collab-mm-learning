"""ISCA baseline network modules and utilities.

Credits for MCC loss and ForeverDataIterator:
  https://github.com/thuml/Transfer-Learning-Library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ MCC loss

def _entropy(predictions: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    return H.sum(dim=1)


class MinimumClassConfusionLoss(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = _entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(1)
        cc = torch.mm((predictions * entropy_weight).t(), predictions)
        cc = cc / torch.sum(cc, dim=1)
        return (torch.sum(cc) - torch.trace(cc)) / num_classes


# ------------------------------------------------------------------ data iterator

class ForeverDataIterator:
    """Infinite iterator that wraps a DataLoader, restarting at StopIteration."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            return next(self.iter)

    def __len__(self):
        return len(self.data_loader)


# ------------------------------------------------------------------ model modules

class ISCADiscriminator(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),       nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),        nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),        nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),        nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),         nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),           nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class ISCAClassifier(nn.Module):
    """Single hidden-layer MLP classifier (matches notebook: num_layers=1 → linear)."""

    def __init__(self, input_size: int, num_classes: int, num_layers: int = 1, hidden_units: int = 256):
        super().__init__()
        if num_layers > 1:
            layers = [nn.Linear(input_size, hidden_units), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
            layers.append(nn.Linear(hidden_units, num_classes))
        else:
            layers = [nn.Linear(input_size, num_classes)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
