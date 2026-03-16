import torch
import torch.nn as nn
from torchvision import models


class MultiViewConcat(nn.Module):

    def __init__(self, output_dim=22):
        super().__init__()

        backbone = models.resnet18(weights=None)

        # modify for grayscale
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # remove final classifier
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # feature dimension of resnet18
        feature_dim = 512

        # final prediction layer after concatenation
        self.fc = nn.Linear(feature_dim * 2, output_dim)

    def forward(self, image_left, image_right):
        
        feat_left = self.feature_extractor(image_left)
        feat_right = self.feature_extractor(image_right)

        feat_left = feat_left.flatten(1)
        feat_right = feat_right.flatten(1)
        
        #concatenate
        combined = torch.cat([feat_left, feat_right], dim=1)

        output = self.fc(combined)

        return output