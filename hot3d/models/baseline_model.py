import torch
import torch.nn as nn
from torchvision import models


#pytorch neural network
class SingleViewBaseline(nn.Module):
    def __init__(self, output_dim=22):
        super().__init__()

        #resNet18 network
        self.backbone = models.resnet18(weights=None)
        
        #expects 3 channels, [3, H, W], we only have 1, [1,H,W]
        #modify input
        #replace the first laye
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, output_dim)

    #defines how we forward input through the model
    def forward(self, x):
        return self.backbone(x)