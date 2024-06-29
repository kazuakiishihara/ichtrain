from models.mednext.MedNextV1 import *
from models.unet3d.model import UNet3D

class MedNeXt_en(nn.Module):
    def __init__(self):
        super().__init__()
        model = MedNeXt(
                    in_channels = 1,                          # input channels
                    n_channels = 32,                          # number of base channels
                    n_classes = 1,                            # number of classes
                    exp_r = 4,                                # Expansion ratio in Expansion Layer
                    kernel_size = 7,                          # Kernel Size in Depthwise Conv. Layer
                    enc_kernel_size = None,                   # (Separate) Kernel Size in Encoder
                    dec_kernel_size = None,                   # (Separate) Kernel Size in Decoder
                    deep_supervision = False,                 # Enable Deep Supervision
                    do_res = False,                           # Residual connection in MedNeXt block
                    do_res_up_down = False,                   # Residual conn. in Resampling blocks
                    checkpoint_style = None,                  # Enable Gradient Checkpointing
                    block_counts = [1,1,1,1,2,1,1,1,1],       # Depth-first no. of blocks per layer 
                    norm_type = 'group',                      # Type of Norm: 'group' or 'layer'
                    dim = '3d'                                # Supports `3d', '2d' arguments
        )
        modules = [
                    model.stem,
                    model.enc_block_0,
                    model.down_0,
                    model.enc_block_1,
                    model.down_1,
                    model.enc_block_2,
                    model.down_2,
                    model.enc_block_3,
                    model.down_3,
                    model.bottleneck,
                ]
        
        self.stem_to_bottleneck_model = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.stem_to_bottleneck_model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Unet3d_en(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        if model is not None:
            model = model
        else:
            model = UNet3D(in_channels=1, out_channels=1)
        self.encoders = model.encoders
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        for encoder in self.encoders:
            x = encoder(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def transfer_Unet3d_en(path):
    model = UNet3D(in_channels=1, out_channels=1)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = Unet3d_en(model=model)
    return model
