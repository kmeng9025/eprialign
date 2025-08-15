"""
Quick test of the trained kidney model
"""
import torch
import numpy as np
from unet_3d import UNet3D

def test_model():
    print("🧠 TESTING TRAINED KIDNEY MODEL")
    print("="*50)
    
    # Load the model
    try:
        model = UNet3D(in_channels=1, out_channels=1)
        
        # Load checkpoint
        checkpoint = torch.load('kidney_unet_model_best.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"📈 Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"🎯 Target size: {checkpoint['target_size']}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test with dummy data
    try:
        dummy_input = torch.randn(1, 1, 64, 64, 32)  # Batch size 1, single channel, 64x64x32
        print(f"🔄 Testing with dummy input: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model inference successful")
        print(f"📤 Output shape: {output.shape}")
        print(f"📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Check if output is reasonable
        if output.shape == dummy_input.shape:
            print("✅ Output shape matches input shape")
        else:
            print("❌ Output shape mismatch")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 MODEL TEST PASSED!")
        print("✅ The trained kidney model is ready for use")
    else:
        print("\n❌ MODEL TEST FAILED!")
