import torch
import torch.nn as nn
import timm


class ResNet(nn.Module):
    def __init__(
            self,
            num_input_channels: int = 12,
            num_output_classes: int = 1,
            dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name="resnet18", # oother option - "resnet10t"
            pretrained=False, 
            in_chans=num_input_channels, 
            features_only=True,
            out_indices=[-1]
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(self.backbone.layer4[1].bn2.num_features, num_output_classes, bias=False) # resnet18
            # torch.nn.Linear(self.backbone.layer4[0].downsample[2].num_features, num_output_classes) # resnet10t
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x)[0])
    
    def activate_dropout(self):
        self.classifier[2].train()

if __name__ == "__main__":
    from torchinfo import summary
    bs = 4
    _ = summary(ResNet(), (bs,12,33,33))
