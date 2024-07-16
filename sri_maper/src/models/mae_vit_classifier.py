from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from sri_maper.src.models.cma_module_pretrain_mae import SSCMALitModule
from sri_maper.src import utils

log = utils.get_pylogger(__name__)


class DummyPatchDropLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches : torch.Tensor):
        return patches, None, None


class CLSClassifier(torch.nn.Module):
    def __init__(self,
        backbone_ckpt: str = None,
        backbone_ckpt_embeddings: str = None,
        backbone_net: torch.nn.Module = None,
        dropout_rate: Optional[List[float]] = [0.5, 0.5, 0.5],
        out_bias: bool = False,
    ) -> None:

        super().__init__()
        if backbone_ckpt_embeddings is not None:
            log.warning("Using pretrained embeddings for faster training and mapping; only likelihoods and uncertainties can be mapped!")
            # using pretrained embeddings:
            self.frozen_embedding = torch.tensor(np.load(backbone_ckpt_embeddings), dtype=torch.float32)
            self.frozen_embedding = torch.nn.Parameter(data=self.frozen_embedding, requires_grad=False)
        else:
            self.frozen_embedding = None
            log.warning("Using backbone network (optionally frozen); likelihoods, uncertainties AND attributions can be mapped!")
            # encoder
            self.frozen_backbone = SSCMALitModule.load_from_checkpoint(backbone_ckpt, net=backbone_net).net.encoder if backbone_ckpt is not None else backbone_net.encoder
            self.frozen_backbone.patch_drop = DummyPatchDropLayer() # prevents input masking
            # freezes backbone
            self.frozen_backbone.requires_grad_(False)

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

    def forward(self, img, col: torch.Tensor, row: torch.Tensor):
        # extracts features
        if self.frozen_embedding is not None:
            features = self.frozen_embedding[row, col]
        else:
            features = self.frozen_backbone(img)[0][:,0,:]
        
        # classfies the CLS token features
        features = self.classifier(features)

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
