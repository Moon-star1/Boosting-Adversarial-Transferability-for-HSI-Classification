import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG11(nn.Module):
    def __init__(self, num_classes=9):
        super(VGG11, self).__init__()
        self.conv_layer1 = self._make_conv_1(103, 64)
        self.conv_layer2 = self._make_conv_1(64, 128)
        self.conv_layer3 = self._make_conv_2(128, 256)
        self.conv_layer4 = self._make_conv_2(256, 512)
        self.conv_layer5 = self._make_conv_2(512, 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)

        )

    def _make_conv_1(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return layer

    def _make_conv_2(self, in_channels, out_channels):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return layer

    def forward(self, x):
        # 32*32 channel == 3
        x = self.conv_layer1(x)
        # 16*16 channel == 64
        x = self.conv_layer2(x)
        # 8*8 channel == 128
        x = self.conv_layer3(x)
        # 4*4 channel == 256
        x = self.conv_layer4(x)
        # 2*2 channel == 512
        x = self.conv_layer5(x)
        # 1*1 channel == 512
        x = x.view(x.size(0), -1)
        # 512
        x = self.classifier(x)
        # 10
        return x
