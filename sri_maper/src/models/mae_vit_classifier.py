from typing import Optional, List, Union

import torch
import torch.nn.functional as F

from sri_maper.src.models.cma_module_pretrain_mae import SSCMALitModule
from sri_maper.src import utils


class DummyPatchDropLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches : torch.Tensor):
        return patches, None, None


class CLSClassifier(torch.nn.Module):
    def __init__(self,
        backbone_ckpt: str = None,
        backbone_net: torch.nn.Module = None,
        freeze_backbone: bool = True,
        dropout_rate: Optional[List[float]] = [0.5, 0.5, 0.5],
        out_bias: bool = False,
    ) -> None:

        super().__init__()
        # encoder
        self.backbone = SSCMALitModule.load_from_checkpoint(backbone_ckpt, net=backbone_net).net.encoder if backbone_ckpt is not None else backbone_net.encoder
        self.backbone.patch_drop = DummyPatchDropLayer() # prevents input masking
        # optionally freezes backbone
        self.backbone.requires_grad_(not freeze_backbone)
        # classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate[0]),
            torch.nn.Linear(backbone_net.enc_dim, backbone_net.enc_dim//2),
            torch.nn.BatchNorm1d(backbone_net.enc_dim//2),

            torch.nn.PReLU(),
            torch.nn.Dropout(p=dropout_rate[1]),
            torch.nn.Linear(backbone_net.enc_dim//2, backbone_net.enc_dim//4),
            torch.nn.BatchNorm1d(backbone_net.enc_dim//4),

            torch.nn.PReLU(),
            torch.nn.Dropout(p=dropout_rate[2]),
            torch.nn.Linear(backbone_net.enc_dim//4, 1, bias=out_bias)
        )

    def forward(self,
                img: torch.Tensor,
                pca_matrix: Union[torch.Tensor, None]=None
        ) -> torch.Tensor:
        img_input = torch.einsum('ijkl,ijm->imkl', img, pca_matrix) if pca_matrix is not None else img
        # extracts features
        features, _, _ = self.backbone(img_input)
        # classfies the CLS token features
        features = self.classifier(features[:,0,:])

        return features

    def activate_dropout(self):
        for m in self.classifier:
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def revert_sync_batchnorm(self):
        # fixes SyncBatchNorm layers if they exist due to multi-GPU training
        self.classifier = utils.revert_sync_batchnorm(self.classifier, torch.nn.modules.batchnorm.BatchNorm1d)

    def contains_sync_batchnorm(self):
        # checks for SynBatchNorms
        return utils.contains_sync_batchnorm(self.classifier)

###########################################################
#        !! TODO: ALL BELOW ARE DEPRECATED !!
###########################################################

class PatchClassifier(torch.nn.Module):
    def __init__(self,
        backbone_ckpt: str = None,
        backbone_net: torch.nn.Module = None,
        freeze_backbone: bool = True,
    ) -> None:

        super().__init__()
        # encoder
        self.backbone = SSCMALitModule.load_from_checkpoint(backbone_ckpt, net=backbone_net).net
        self.backbone.encoder.patch_drop = DummyPatchDropLayer() # prevents input masking
        # optionally freezes backbone
        self.backbone.requires_grad_(not freeze_backbone)
        # classifier
        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(self.backbone.enc_dim, 1)

    def forward(self, img):
        # extracts features, removing CLS token ->  [batch_sisze, num_of_patches, emb_dim]
        features, _, _ = self.backbone.encoder(img)[:,1:,:]
        # classifies the patch features
        features = self.pooling(features.transpose(1, 2)).squeeze(-1) # [batch_size, emb_dim]
        features = self.classifier(features)

        return features

##############################################################################################################################

class AttnPatchClassifier(torch.nn.Module):
    def __init__(self,
        backbone_ckpt: str = None,
        backbone_net: torch.nn.Module = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        # encoder
        self.backbone = SSCMALitModule.load_from_checkpoint(backbone_ckpt, net=backbone_net).net
        self.backbone.encoder.patch_drop = DummyPatchDropLayer() # prevents input masking
        # optionally freezes backbone
        self.backbone.requires_grad_(not freeze_backbone)
        # classifier
        self.attention = torch.nn.MultiheadAttention(self.backbone.enc_dim, 1, batch_first=True)
        feat_dim = self.backbone.enc_dim * ((self.backbone.image_size // self.backbone.patch_size) ** 2 + 1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(feat_dim, 1)
        )

    def forward(self, img):
        # extracts features ->  [batch_sisze, num_of_patches+1, emb_dim]
        features, _, _ = self.backbone.encoder(img)
        # classifer -> [batch_size, num_of_patches * emb_dim]
        features, _ = self.attention(query=features, key=features, value=features)
        features = self.classifier(features)

        return features

##############################################################################################################################

class ConvPatchClassifier(torch.nn.Module):
    def __init__(self,
            backbone_ckpt: str = None,
            backbone_net: torch.nn.Module = None,
            num_filters: int = 64,
            kernel_size: int = 3,
            freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        # encoder
        self.backbone = SSCMALitModule.load_from_checkpoint(backbone_ckpt, net=backbone_net).net
        self.backbone.encoder.patch_drop = DummyPatchDropLayer() # prevents input masking
        # optionally freezes backbone
        self.backbone.requires_grad_(not freeze_backbone)
        # classifier
        self.conv = torch.nn.Conv1d(in_channels=self.backbone.enc_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.classifier = torch.nn.Linear(num_filters, 1)

    def forward(self, img):
        # extracts features, removing CLS token ->  [batch_sisze, num_of_patches, emb_dim]
        features, _, _ = self.backbone.encoder(img)[:,1:,:]
        # classifier
        features = F.relu(self.conv(features.transpose(1, 2))) # -> [batch_size, num_filters, num_patches]
        features = self.pool(features).squeeze(-1) # -> [batch_size, num_filters]
        features = self.classifier(features)

        return features
