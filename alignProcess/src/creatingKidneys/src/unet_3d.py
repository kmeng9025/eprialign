import torch
import torch.nn as nn

class UNet3D(nn.Module):
    """3D U-Net for kidney segmentation matching Modal training architecture"""
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        
        for feature in features:
            self.encoder.append(self._make_conv_block(in_channels, feature))
            self.encoder_pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._make_conv_block(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.decoder_upconvs = nn.ModuleList()
        
        for feature in reversed(features):
            self.decoder_upconvs.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._make_conv_block(feature * 2, feature))
        
        # Final layer
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def _make_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder, pool in zip(self.encoder, self.encoder_pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        for idx, (upconv, decoder) in enumerate(zip(self.decoder_upconvs, self.decoder)):
            x = upconv(x)
            skip_connection = skip_connections[idx]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='trilinear', align_corners=False)
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = decoder(concat_skip)
        
        # Final output
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x
