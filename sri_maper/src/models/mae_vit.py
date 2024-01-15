import torch
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from einops.layers.torch import Rearrange


class PatchDropLayer(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio
    
    def forward(self, patches : torch.Tensor):
        B, L, D = patches.shape  # batch, length, dim
        len_keep = int(L * (1 - self.ratio))
        
        # sorts noise for each sample
        noise = torch.rand(B, L, device=patches.device)
        shuffle = torch.argsort(noise, dim=1)
        restore = torch.argsort(shuffle, dim=1)

        # keeps the first subset
        keep = shuffle[:, :len_keep]
        masked_patches = torch.gather(patches, dim=1, index=keep.unsqueeze(-1).repeat(1, 1, D))

        # generates the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=patches.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=restore)

        return masked_patches, mask, restore


class MAE_Encoder(torch.nn.Module):
    def __init__(
        self,
        image_size:    int = 33,
        patch_size:    int = 11,
        input_dim:     int = 73,
        emb_dim:       int = 192,
        num_layer:     int = 12,
        num_head:      int = 3,
        mask_ratio:    float = 0.0,
    ) -> None:
        
        super().__init__()

        # inits learned CLS token
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # inits learned position encoding
        self.pos_encoding = torch.nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, emb_dim))
        # inits learned patch embedding
        self.patch_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_dim, 
                out_channels=emb_dim, 
                kernel_size=patch_size, 
                stride=patch_size
            ),
            torch.nn.Flatten(start_dim=2)
        )
        # inits masking layer
        self.patch_drop = PatchDropLayer(mask_ratio)
        # inits feature extractor backbone
        self.backbone = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        # inits normalization
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_encoding, std=.02)

    def forward(self, img):
        # embeds img (B, C, H, W) -> (B, L, C')
        patches = self.patch_embedding(img).transpose(1,2)
        # adds position learned encoding
        patches = patches + self.pos_encoding
        # drops mask_ratio of the patches (B, L, C') -> (B, L', C')
        patches, mask, restore = self.patch_drop(patches)
        # inserts the cls token
        patches = torch.cat([self.cls_token.expand(patches.shape[0], -1, -1), patches], dim=1)
        # extracts features
        features = self.layer_norm(self.backbone(patches))
        # returns features (B, L', C'), mask (B, L), and restore (B, L)
        return features, mask, restore


class MAE_Decoder(torch.nn.Module):
    def __init__(
        self,
        image_size:    int = 33,
        patch_size:    int = 11,
        enc_dim:       int = 192,
        dec_dim:       int = 192,
        output_dim:    int = 73,
        num_layer:     int = 4,
        num_head:      int = 3,
    ) -> None:
        
        super().__init__()

        # inits projection to decoder dim
        self.decoder_proj = torch.nn.Linear(enc_dim, dec_dim, bias=True)
        # inits mask token
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, dec_dim))
        # inits pos encoding
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, dec_dim))
        # inits backbone feature extractor
        self.backbone = torch.nn.Sequential(*[Block(dec_dim, num_head) for _ in range(num_layer)])
        # inits normalization
        self.layer_norm = torch.nn.LayerNorm(dec_dim)
        # inits decoder prediction head
        self.head = torch.nn.Linear(dec_dim, output_dim * patch_size ** 2)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, restore):
        # projects enc dim to dec dim (B, L', C') -> (B, L', C'')
        features = self.decoder_proj(features)
        # inserts mask tokens into unshuffled sequence
        mask_tokens = self.mask_token.repeat(features.shape[0], restore.shape[1]+1 - features.shape[1], 1)
        features_cls = features[:, 0, :].clone().unsqueeze(1)
        features = torch.cat([features[:, 1:, :], mask_tokens], dim=1)
        features = torch.gather(features, dim=1, index=restore.unsqueeze(-1).repeat(1,1, features.shape[2]))
        # adds CLS feature
        features = torch.cat([features_cls, features], dim=1)
        # adds position learned encoding
        features = features + self.pos_embedding
        # extracts features, removing CLS
        features = self.layer_norm(self.backbone(features))[:, 1:, :]
        # returns predict patches
        return self.head(features)


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size:        int = 33,
                 patch_size:        int = 11,
                 input_dim:         int = 73,
                 enc_dim:           int = 192,
                 dec_dim:           int = 192,
                 output_dim:        int = 73,
                 encoder_layer:     int = 12,
                 encoder_head:      int = 3,
                 decoder_layer:     int = 4,
                 decoder_head:      int = 3,
                 mask_ratio:        float = 0.0,
                 ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.enc_dim = enc_dim
        # inits MAE enoder
        self.encoder = MAE_Encoder(image_size, patch_size, input_dim, enc_dim, encoder_layer, encoder_head, mask_ratio)
        # inits MAE decoder
        self.decoder = MAE_Decoder(image_size, patch_size, enc_dim, dec_dim, output_dim, decoder_layer, decoder_head)
        # inits layer to merge patches into image
        self.patch2img = Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h= image_size // patch_size)

    def forward(self, img):
        # encodes image patches
        features, mask, restore = self.encoder(img)
        # reconstructs encoded image patches
        predicted_img = self.decoder(features,  restore)
        # returns combined patches into images
        return self.patch2img(predicted_img), self.patch2img(mask.unsqueeze(-1).repeat(1, 1, predicted_img.shape[-1]))