import torch.nn as nn
import torchvision.models as models


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        # Load the pretrained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace the fully connected layer for binary classification (2 classes)
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)