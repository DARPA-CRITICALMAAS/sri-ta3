import torch
import timm
import numpy as np
from typing import Optional

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import torch.nn.functional as F

from models.mae_vit import MAE_Encoder

class cls_token_classifier(torch.nn.Module):
    def __init__(self, 
                 encoder : Optional[torch.nn.Module] = MAE_Encoder, 
        ) -> None:
        super().__init__()
        ### Encoder ###
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        ###############

        self.ff = torch.nn.Linear(encoder.emb_dim, 1)
           
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches)) # -> [batch_size, 1+num_of_patches, emb_dim]; +1 - CLS token

        features = features[:,0,:] # extract the CLS token -> [batch_size, emb_dim]

        features = self.ff(features)

        return features

##############################################################################################################################

class patch_classifier_w_AdaptiveAvgPool1d_across_patch_dim(torch.nn.Module):
    def __init__(self, 
                 encoder : Optional[torch.nn.Module] = MAE_Encoder, 
        ) -> None:
        super().__init__()
        ### Encoder ###
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        ###############

        self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        self.ff = torch.nn.Linear(encoder.emb_dim, 1)
           
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches)) # -> [batch_size, 1+num_of_patches, emb_dim]; +1 - CLS token

        features = features[:,1:,:] # remove CLS token  [batch_sisze, num_of_patches, emb_dim]

        features = self.pooling(features.transpose(1, 2)).squeeze(-1) # [batch_size, emb_dim]
        features = self.ff(features)

        return features

##############################################################################################################################

class SelfAttention(torch.nn.Module):
    def __init__(self, emb_dim):
        super(SelfAttention, self).__init__()
        self.query = torch.nn.Linear(emb_dim, emb_dim)
        self.key = torch.nn.Linear(emb_dim, emb_dim)
        self.value = torch.nn.Linear(emb_dim, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.emb_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, V)

class patch_classifier_w_SelfAttn(torch.nn.Module):
    def __init__(self, 
                 encoder : Optional[torch.nn.Module] = MAE_Encoder, 
        ) -> None:
        super().__init__()
        ### Encoder ###
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        ###############

        self.attention = SelfAttention(encoder.emb_dim)
        self.ff = torch.nn.Linear(encoder.emb_dim, 1)
           
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches)) # -> [batch_size, 1+num_of_patches, emb_dim]; +1 - CLS token

        features = features[:,1:,:] # remove CLS token -> [batch_size, num_of_patches, emb_dim]

        features = self.attention(features) # [batch_size, num_of_patches, emb_dim]
        features = features.mean(dim=1) # [batch_size, emb_dim]
        features = self.ff(features)

        return features
    
##############################################################################################################################

class patch_classifier_w_Conv(torch.nn.Module):
    def __init__(self, 
                 encoder : Optional[torch.nn.Module] = MAE_Encoder,  
                 num_filters: int = 64,
                 kernel_size: int = 3,
        ) -> None:
        super().__init__()
        ### Encoder ###
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        ###############

        self.conv = torch.nn.Conv1d(in_channels=encoder.emb_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.ff = torch.nn.Linear(num_filters, 1)
           
    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches)) # -> [batch_size, 1+num_of_patches, emb_dim]; +1 - CLS token

        features = features[:,1:,:] # remove CLS token -> [batch_size, num_of_patches, emb_dim]

        features = F.relu(self.conv(features.transpose(1, 2))) # -> [batch_size, num_filters, num_patches]
        features = self.pool(features).squeeze(-1) # -> [batch_size, num_filters]
        features = self.ff(features)

        return features
    
##############################################################################################################################