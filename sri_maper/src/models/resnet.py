import torch
import torch.nn as nn
import timm
from typing import Literal, Union

from sri_maper.src import utils


class ResNet(nn.Module):
    def __init__(
            self,
            num_input_channels: int = 12,
            num_output_classes: int = 1,
            dropout_rate: float = 0.5,
            backbone_name: Literal["resnet18", "resnet10t"] = "resnet18",
            out_bias: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name=backbone_name, # other option - "resnet10t"
            pretrained=False,
            in_chans=num_input_channels,
            features_only=True,
            out_indices=[-1]
        )

        if backbone_name == "resnet18":
            backbone_features = self.backbone.layer4[1].bn2.num_features
        elif backbone_name == "resnet10t":
            backbone_features = self.backbone.layer4[0].downsample[2].num_features
        else:
            raise ValueError(f"backbone_name must be one of ['resnet18', 'resnet10t'], got {backbone_name}")


        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(backbone_features, num_output_classes, bias=out_bias)
        )

    def forward(self,
                img: torch.Tensor,
                pca_matrix: Union[torch.Tensor, None]=None
        ) -> torch.Tensor:
        img_input = torch.einsum('ijkl,ijm->imkl', img, pca_matrix) if pca_matrix is not None else img
        return self.classifier(self.backbone(img_input)[0])

    def activate_dropout(self):
        for m in self.classifier:
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

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
