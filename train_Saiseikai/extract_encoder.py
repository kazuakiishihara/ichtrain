import torch
from torch import nn

from models.unet3d.model import UNet3D

class Unet3d_en(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        if model is not None:
            model = model
        else:
            model = UNet3D(in_channels=1, out_channels=1)
        self.encoders = model.encoders
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        x = self.avgpool(x)
        z512 = x.view(x.size(0), -1)
        z128 = self.fc1(z512)
        x = self.relu(z128)
        x = self.fc2(x)
        return x, z512, z128

def transfer_Unet3d_en(path):
    model = UNet3D(in_channels=1, out_channels=1)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = Unet3d_en(model=model)
    return model
