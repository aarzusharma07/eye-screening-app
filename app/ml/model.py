import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained=True):
    # Load pre-trained EfficientNet-B3
    if pretrained:
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b3()

    # Unfreeze layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Replace final fully connected classifier layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model
