import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Identity()
        self.resnet = resnet
        self.output_dim = 512

        if freeze_base:
            for name, param in self.resnet.named_parameters():
                if "layer4" not in name:
                    param.requires_grad = False

    def forward(self, image):
        return self.resnet(image)