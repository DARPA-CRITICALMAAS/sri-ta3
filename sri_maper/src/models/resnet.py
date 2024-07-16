import torch
import torch.nn as nn
import timm

from sri_maper.src import utils


class ResNet(nn.Module):
    def __init__(
            self,
            num_input_channels: int = 12,
            num_output_classes: int = 1,
            dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name="resnet10t", # oother option - "resnet10t"
            pretrained=False,
            in_chans=num_input_channels,
            features_only=True,
            out_indices=[-1]
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(p=dropout_rate),
            # torch.nn.Linear(self.backbone.layer4[1].bn2.num_features, num_output_classes, bias=False) # resnet18
            torch.nn.Linear(self.backbone.layer4[0].downsample[2].num_features, num_output_classes) # resnet10t
        )
        self.frozen_embedding = None

    def forward(self, x: torch.Tensor, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x)[0])

    def activate_dropout(self):
        self.classifier[2].train()

    def revert_sync_batchnorm(self):
        # fixes SyncBatchNorm layers if they exist due to multi-GPU training
        self.backbone = utils.revert_sync_batchnorm(self.backbone, torch.nn.modules.batchnorm.BatchNorm2d)

    def contains_sync_batchnorm(self):
        # checks for SynBatchNorms
        return utils.contains_sync_batchnorm(self.backbone)


if __name__ == "__main__":
    from torchinfo import summary
    bs = 4
    _ = summary(ResNet(), (bs,12,33,33))
