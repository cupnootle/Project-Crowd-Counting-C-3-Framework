import torch.nn as nn
import torch
from torchvision import models
import torch.nn.functional as F

class CSRNet_aleatoric(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet_aleatoric, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())

        self.fc_pred = nn.Conv2d(64, 1, kernel_size=1)
        self.fc_logvar = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        pred = self.fc_pred(x)
        pred = F.softplus(pred)
        pred = F.upsample(pred, scale_factor=8)
        logvar = F.softplus(self.fc_logvar(x)) #softplus is a smooth approximation to ReLU
        logvar = F.upsample(logvar, scale_factor=8)
        return pred, logvar #the model now outputs both a prediction and a logvar

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(
                in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate
            )
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)