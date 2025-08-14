"""
Model definitions for fiducial detection
This module contains only the model classes without training code
"""

import torch
import torch.nn as nn

# Define 3D U-Net model for fiducial detection
class FiducialUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(FiducialUNet3D, self).__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = conv_block(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)
        
        # Output layer
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)
        
    def forward(self, x, return_logits=False):
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
        
        # Output
        logits = self.final(dec1)
        
        if return_logits:
            return logits
        else:
            return torch.sigmoid(logits)

# Combined Dice + Focal Loss
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets, smooth=1.0):
        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        
        intersection = (inputs * targets).sum(dim=1)
        total = inputs.sum(dim=1) + targets.sum(dim=1)
        
        # Dice coefficient
        dice = (2. * intersection + smooth) / (total + smooth)
        
        # Focal weighting for hard examples
        dice_loss = 1 - dice
        focal_weight = self.alpha * (dice_loss ** self.gamma)
        
        return (focal_weight * dice_loss).mean()

# Tversky Loss (good for imbalanced data)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, inputs, targets, smooth=1.0):
        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        
        TP = (inputs * targets).sum(dim=1)
        FP = ((1-targets) * inputs).sum(dim=1)
        FN = (targets * (1-inputs)).sum(dim=1)
        
        tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        
        return (1 - tversky).mean()

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_dice_loss = FocalDiceLoss(alpha=1.0, gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        
    def forward(self, logits, targets):
        # Apply sigmoid to logits for dice and tversky losses
        probs = torch.sigmoid(logits)
        
        # Calculate individual losses
        bce = self.bce_loss(logits, targets)
        focal_dice = self.focal_dice_loss(probs, targets)
        tversky = self.tversky_loss(probs, targets)
        
        # Combine losses with weights
        total_loss = 0.3 * bce + 0.4 * focal_dice + 0.3 * tversky
        
        return total_loss, {
            'bce': bce.item(),
            'focal_dice': focal_dice.item(),
            'tversky': tversky.item(),
            'total': total_loss.item()
        }
