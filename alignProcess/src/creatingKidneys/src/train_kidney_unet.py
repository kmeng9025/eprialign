import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import time
import pickle
from scipy.ndimage import zoom

# Define input/output paths
DATA_DIR = '.'
MODEL_OUT = 'kidney_unet_model.pth'
BEST_MODEL_OUT = 'kidney_unet_model_best.pth'
KIDNEY_DATA_PATH = 'kidney_training_data.pkl'

# Training parameters
BATCH_SIZE = 2 if torch.cuda.is_available() else 1  # Small batch for 3D data
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2
TARGET_SIZE = (64, 64, 32)  # Resize to manageable size

# Define 3D U-Net model for kidney segmentation
class KidneyUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(KidneyUNet3D, self).__init__()
        
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

# Enhanced Dice Loss for kidney segmentation
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets, smooth=1.0):
        # Flatten tensors
        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum(dim=1)
        total = inputs.sum(dim=1) + targets.sum(dim=1)
        
        # Dice coefficient
        dice = (2. * intersection + smooth) / (total + smooth)
        
        # Focal weighting for hard examples
        dice_loss = 1 - dice
        focal_weight = self.alpha * (dice_loss ** self.gamma)
        
        return (focal_weight * dice_loss).mean()

# Tversky Loss (good for imbalanced data like kidneys)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, inputs, targets, smooth=1.0):
        inputs = inputs.contiguous().view(inputs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(dim=1)
        FP = ((1-targets) * inputs).sum(dim=1)
        FN = (targets * (1-inputs)).sum(dim=1)
        
        tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)
        return 1 - tversky.mean()

def calculate_metrics(pred, target, threshold=0.5):
    """Calculate various metrics for evaluation"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten for calculation
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Basic metrics
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    # Calculate metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def resize_volume(volume, target_shape):
    """Resize 3D volume to target shape"""
    if volume.shape == target_shape:
        return volume
    
    zoom_factors = [t/s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, zoom_factors, order=1)

def load_and_prepare_kidney_data():
    """Load kidney training data and prepare for PyTorch"""
    print("📂 Loading kidney training data...")
    
    if not os.path.exists(KIDNEY_DATA_PATH):
        print("❌ Kidney training data not found!")
        print("Please run extract_training_arrays.py first.")
        return None, None
    
    # Load the data
    with open(KIDNEY_DATA_PATH, 'rb') as f:
        training_data = pickle.load(f)
    
    mri_volumes = training_data['mri_volumes']
    kidney_masks = training_data['kidney_masks']
    metadata = training_data['metadata']
    
    print(f"📊 Loaded {len(mri_volumes)} kidney training volumes")
    
    # Prepare data for training
    X_list = []
    Y_list = []
    
    for i, (mri, mask, meta) in enumerate(zip(mri_volumes, kidney_masks, metadata)):
        print(f"  Processing volume {i+1}: {meta['file']} - {mri.shape}")
        
        # Resize to target size
        mri_resized = resize_volume(mri, TARGET_SIZE)
        mask_resized = resize_volume(mask.astype(np.float32), TARGET_SIZE)
        
        # Ensure mask is binary after resizing
        mask_resized = (mask_resized > 0.5).astype(np.float32)
        
        # Add channel dimension: (1, D, H, W)
        mri_with_channel = mri_resized[np.newaxis, ...]
        mask_with_channel = mask_resized[np.newaxis, ...]
        
        X_list.append(mri_with_channel)
        Y_list.append(mask_with_channel)
        
        kidney_voxels = np.sum(mask_resized)
        total_voxels = mask_resized.size
        coverage = (kidney_voxels / total_voxels) * 100
        print(f"    Resized to {TARGET_SIZE}, kidney coverage: {coverage:.2f}%")
    
    # Convert to numpy arrays
    X = np.array(X_list)  # (N, 1, D, H, W)
    Y = np.array(Y_list)  # (N, 1, D, H, W)
    
    print(f"📊 Final dataset:")
    print(f"   Input shape: {X.shape}")
    print(f"   Target shape: {Y.shape}")
    print(f"   Input range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   Target range: [{Y.min():.3f}, {Y.max():.3f}]")
    
    # Check for positive samples
    positive_samples = (Y.sum(axis=(1,2,3,4)) > 0).sum()
    print(f"   Samples with kidneys: {positive_samples}/{len(Y)}")
    
    return X, Y

def main():
    """Main training function"""
    print("🏥 KIDNEY SEGMENTATION U-NET TRAINING")
    print("="*50)
    
    # Load and prepare data
    X, Y = load_and_prepare_kidney_data()
    if X is None:
        return
    
    if len(X) < 2:
        print("❌ Need at least 2 samples for training!")
        return
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    
    # Create dataset and split
    dataset = TensorDataset(X_tensor, Y_tensor)
    
    if len(dataset) >= 5:
        train_size = max(1, int((1 - VALIDATION_SPLIT) * len(dataset)))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        # Use all data for training if too few samples
        train_dataset = dataset
        val_dataset = dataset
        print("⚠️  Using all data for both training and validation (too few samples)")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"📈 Training samples: {len(train_dataset)}")
    print(f"📈 Validation samples: {len(val_dataset)}")
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔢 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Enable mixed precision training for better GPU utilization
        try:
            from torch.amp import autocast, GradScaler
            scaler = GradScaler('cuda')
            use_amp = True
        except ImportError:
            print("⚠️  Mixed precision not available")
            use_amp = False
    else:
        print("💻 Using CPU - training will be slower")
        use_amp = False
    
    model = KidneyUNet3D().to(device)
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    focal_dice_loss = FocalDiceLoss()
    tversky_loss = TverskyLoss()
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Resume training if model exists
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if os.path.exists(MODEL_OUT):
        choice = input("⚠️ Model weights found. Resume training? (y/n): ").strip().lower()
        if choice == 'y':
            checkpoint = torch.load(MODEL_OUT, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
                print(f"✅ Resumed from epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                print("✅ Loaded model weights (legacy format)")
        else:
            print("🆕 Starting training from scratch")
    else:
        print("🆕 No existing model found. Training from scratch")
    
    # Training loop
    print(f"\n🚀 Starting training for {NUM_EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision if available
            if use_amp:
                with autocast('cuda'):
                    logits = model(batch_x, apply_sigmoid=False)
                    output = torch.sigmoid(logits)
                    
                    # Combined loss
                    loss_bce = bce_loss(logits, batch_y)
                    loss_focal_dice = focal_dice_loss(output, batch_y)
                    loss_tversky = tversky_loss(output, batch_y)
                    
                    # Weighted combination (emphasize dice for kidneys)
                    total_loss = 0.2 * loss_bce + 0.5 * loss_focal_dice + 0.3 * loss_tversky
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_x, apply_sigmoid=False)
                output = torch.sigmoid(logits)
                
                # Combined loss
                loss_bce = bce_loss(logits, batch_y)
                loss_focal_dice = focal_dice_loss(output, batch_y)
                loss_tversky = tversky_loss(output, batch_y)
                
                # Weighted combination (emphasize dice for kidneys)
                total_loss = 0.2 * loss_bce + 0.5 * loss_focal_dice + 0.3 * loss_tversky
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(output.cpu(), batch_y.cpu())
                for key in train_metrics:
                    train_metrics[key] += metrics[key]
        
        # Average training metrics
        avg_train_loss = train_loss / len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                logits = model(batch_x, apply_sigmoid=False)
                output = torch.sigmoid(logits)
                
                # Calculate validation loss
                loss_bce = bce_loss(logits, batch_y)
                loss_focal_dice = focal_dice_loss(output, batch_y)
                loss_tversky = tversky_loss(output, batch_y)
                
                total_loss = 0.2 * loss_bce + 0.5 * loss_focal_dice + 0.3 * loss_tversky
                val_loss += total_loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(output.cpu(), batch_y.cpu())
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
        
        # Average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print progress
        current_time = time.time()
        elapsed = current_time - start_time
        
        print(f"📈 Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val F1: {val_metrics['f1']:.3f} | Val IoU: {val_metrics['iou']:.3f} | "
              f"Time: {elapsed/60:.1f}min")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Save comprehensive checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_metrics': val_metrics,
                'target_size': TARGET_SIZE
            }
            
            torch.save(checkpoint, BEST_MODEL_OUT)
            print(f"💾 Best model saved (Val Loss: {best_val_loss:.4f})")
    
    # Save final model
    final_checkpoint = {
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'target_size': TARGET_SIZE
    }
    
    torch.save(final_checkpoint, MODEL_OUT)
    print(f"💾 Final model saved to {MODEL_OUT}")
    
    # Create training plots
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Kidney Segmentation Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs_range = range(len(train_losses))
    if len(val_losses) > 0:
        plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kidney_training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Best model saved to: {BEST_MODEL_OUT}")
    print(f"💾 Final model saved to: {MODEL_OUT}")

if __name__ == "__main__":
    main()
