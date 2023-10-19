import torch
import torch.nn as nn

class CNN3DNet(nn.Module):
    def __init__(
            self, 
            w: int = 33,
            h: int = 33,
            d: int = 23,
            num_input_channels: int = 1,
            num_output_classes: int = 1,
    ) -> None:
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(num_input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.PReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.PReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2) #nn.AvgPool2d
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * (w//4) * (h//4) * (d//4), 64)
        self.relu3 = nn.PReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization
        
        self.fc2 = nn.Linear(64, num_output_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the output
        
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        
        return x.squeeze()

if __name__ == "__main__":
    _ = CNN3DNet()
