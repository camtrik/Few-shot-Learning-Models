import torch 
import torch.nn as nn 
from .make_models import register 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(residual)
        out = self.relu(out)
        out = self.maxpool(out)

        return out

class ResNet12(nn.Module):
    def __init__(self, channels):
        super(ResNet12, self).__init__()
        self.in_channels = 3

        # 4 * 3 conv, 12 layers
        self.layer1 = self.make_layers(channels[0])
        self.layer2 = self.make_layers(channels[1])
        self.layer3 = self.make_layers(channels[2])
        self.layer4 = self.make_layers(channels[3])

        self.out_dim = channels[3]

        # use kaiming initialization to initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def make_layers(self, cur_channels):
        block = ResidualBlock(self.in_channels, cur_channels)
        self.in_channels = cur_channels 
        return block 

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # (B, C, H, W) -> (B, C, H * W), then compute mean of dim=2
        out = out.view(out.shape[0], out.shape[1], -1).mean(dim=2)

        return out

@register('resnet12')
def resnet12():
    return ResNet12([64, 128, 256, 512])
