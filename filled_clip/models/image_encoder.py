import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
import timm


class ViT(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
        'resnet50', True, num_classes=0, global_pool="avg"
    )
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)
