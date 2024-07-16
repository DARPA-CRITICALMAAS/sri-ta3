from typing import Optional, List

import torch
import torch.nn as nn
import timm

from sri_maper.src import utils


class ANN(nn.Module):
    def __init__(
            self,
            num_input_channels: int = 12,
            image_size: int = 5,
            num_output_classes: int = 1,
            dropout_rate: Optional[List[float]] = [0.0, 0.25, 0.25],
            out_bias: bool = False,
    ) -> None:
        super().__init__()

        # classifier
        self.image_size = image_size
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate[0]),
            torch.nn.Linear(num_input_channels, num_input_channels//2),
            torch.nn.BatchNorm1d(num_input_channels//2),

            torch.nn.PReLU(),
            torch.nn.Dropout(p=dropout_rate[1]),
            torch.nn.Linear(num_input_channels//2, num_input_channels//4),
            torch.nn.BatchNorm1d(num_input_channels//4),

            torch.nn.PReLU(),
            torch.nn.Dropout(p=dropout_rate[2]),
            torch.nn.Linear(num_input_channels//4, num_output_classes, bias=out_bias)
        )
        self.frozen_embedding = None

    def forward(self, x: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        x = x[:, :, self.image_size//2, self.image_size//2]
        return self.classifier(x)

    def activate_dropout(self):
        for m in self.classifier:
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def revert_sync_batchnorm(self):
        # fixes SyncBatchNorm layers if they exist due to multi-GPU training
        self.backbone = utils.revert_sync_batchnorm(self.classifier, torch.nn.modules.batchnorm.BatchNorm1d)

    def contains_sync_batchnorm(self):
        # checks for SynBatchNorms
        return utils.contains_sync_batchnorm(self.classifier)

