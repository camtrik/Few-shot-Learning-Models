import torch.nn as nn 
from .make_models import register_model

# Define the ConvNet4 model
@register_model('convnet4')
class ConvNet4(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ConvNet4, self).__init__()
        self.net = nn.Sequential(
            self.conv_block(x_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim*2),
            self.conv_block(hid_dim*2, hid_dim*2),
            self.conv_block(hid_dim*2, z_dim)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

