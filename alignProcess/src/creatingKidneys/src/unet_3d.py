import torch
import torch.nn as nn

class UNet3D(nn.Module):
    """3D U-Net model for kidney segmentation"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Encoder (smaller for kidney data)
        self.enc1 = conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(64, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = conv_block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = conv_block(32, 16)
        
        # Final layer
        self.final = nn.Conv3d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, apply_sigmoid=True):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        logits = self.final(dec1)
        
        if apply_sigmoid:
            return self.sigmoid(logits)
        else:
            return logits
