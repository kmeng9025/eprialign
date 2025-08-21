"""
Simple AI Kidney Detection - Back to Basics
Just load AI model, predict kidneys, create slaves, done.
"""
import os
import sys
import numpy as np
import torch
import scipy.io as sio
from datetime import datetime
from scipy.ndimage import zoom, binary_erosion, binary_dilation
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from models.unet3d import UNet3D

class SimpleAIKidneyDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained kidney detection model"""
        model_path = 'kidney_unet_model_best.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"🧠 Loading AI model: {model_path}")
        
        # Initialize model
        self.model = UNet3D(in_channels=1, out_channels=1)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ AI model loaded successfully")
    
    def preprocess_mri(self, mri_data):
        """Simple preprocessing like in training"""
        # Normalize to 0-1 using percentiles
        p1, p99 = np.percentile(mri_data, [1, 99])
        mri_normalized = (mri_data - p1) / (p99 - p1)
        mri_normalized = np.clip(mri_normalized, 0, 1)
        
        # Resize to model input size (64x64x32)
        target_shape = (64, 64, 32)
        zoom_factors = [target_shape[i] / mri_data.shape[i] for i in range(3)]
        mri_resized = zoom(mri_normalized, zoom_factors, order=1)
        
        return mri_resized, zoom_factors
    
    def predict_kidneys(self, mri_data):
        """Run AI prediction on MRI data"""
        # Preprocess
        mri_processed, zoom_factors = self.preprocess_mri(mri_data)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(mri_processed).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Run prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.sigmoid(prediction)
        
        # Convert back to numpy and resize to original shape
        pred_np = prediction.squeeze().cpu().numpy()
        
        # Resize back to original shape
        inverse_zoom = [1.0 / z for z in zoom_factors]
        kidney_mask = zoom(pred_np, inverse_zoom, order=1)
        
        # Make sure we have the exact original shape
        kidney_mask = kidney_mask[:mri_data.shape[0], :mri_data.shape[1], :mri_data.shape[2]]
        
        # Apply threshold
        threshold = 0.5  # Simple threshold
        kidney_mask = (kidney_mask > threshold).astype(np.uint8)
        
        return kidney_mask
    
    def process_file(self, input_file):
        """Process a single .mat file"""
        print(f"🤖 Processing: {os.path.basename(input_file)}")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        kidneys_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(kidneys_dir, 'inference', f'ai_kidneys_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the .mat file
        print("📂 Loading data...")
        data = sio.loadmat(input_file, struct_as_record=False, squeeze_me=True)
        
        if 'images' not in data:
            raise ValueError("No 'images' field found")
        
        images = data['images']
        
        # Find MRI images - look for 350x350xN shape
        mri_images = []
        
        for i in range(len(images)):
            img = images[i]
            if hasattr(img, 'data') and img.data is not None:
                if hasattr(img.data, 'shape') and len(img.data.shape) == 3:
                    shape = img.data.shape
                    
                    # Get image name
                    image_name = ""
                    if hasattr(img, 'Name') and img.Name is not None:
                        if isinstance(img.Name, str):
                            image_name = img.Name
                        else:
                            try:
                                image_name = ''.join(chr(c) for c in img.Name.flatten() if c != 0)
                            except:
                                image_name = f"Image_{i}"
                    
                    # Check if it's MRI (350x350xN or has "MRI" in name)
                    if (shape[0] == 350 and shape[1] == 350) or "mri" in image_name.lower():
                        mri_images.append({
                            'index': i,
                            'data': img.data,
                            'name': image_name,
                            'shape': shape
                        })
        
        if not mri_images:
            raise ValueError("No MRI images found")
        
        print(f"✅ Found {len(mri_images)} MRI image(s)")
        for mri in mri_images:
            print(f"   - {mri['name']}: {mri['shape']}")
        
        # Process each MRI and create kidney slaves
        for mri in mri_images:
            print(f"\n🧠 Processing {mri['name']}...")
            
            # Run AI prediction
            kidney_mask = self.predict_kidneys(mri['data'])
            
            # Count kidneys
            num_kidneys = np.sum(kidney_mask > 0)
            coverage = (num_kidneys / kidney_mask.size) * 100
            
            print(f"   🎯 Kidney pixels detected: {num_kidneys}")
            print(f"   📊 Coverage: {coverage:.2f}%")
            
            # Create kidney slave in the original images array
            # Find a free slave index
            slave_idx = len(images)
            
            # Create new slave structure
            slave = type('Slave', (), {})()
            slave.data = kidney_mask.astype(np.uint8)
            slave.Name = f"KidneyAI_{mri['name']}"
            slave.Type = 'binary'
            slave.Master = mri['index']
            
            # Add to images array
            images_list = list(images) if hasattr(images, '__len__') else [images]
            images_list.append(slave)
            
            # Update the data structure
            data['images'] = np.array(images_list, dtype=object)
            
            print(f"   ✅ Created kidney slave: {slave.Name}")
        
        # Save the result
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_WITH_AI_KIDNEYS.mat")
        sio.savemat(output_file, data, format='5')
        
        print(f"\n✅ SUCCESS! Output saved to:")
        print(f"   📁 {output_file}")
        
        return output_file

def main():
    print("🤖 SIMPLE AI KIDNEY DETECTION")
    print("=" * 50)
    
    # Get input file
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]
    else:
        # Look for .mat files in current directory
        input_files = [f for f in os.listdir('.') if f.endswith('.mat')]
        if not input_files:
            print("❌ No .mat files found in current directory")
            print("Usage: python ai_kidney_detection_simple.py <file.mat>")
            return
    
    # Initialize detector
    detector = SimpleAIKidneyDetector()
    
    # Process each file
    for input_file in input_files:
        try:
            detector.process_file(input_file)
        except Exception as e:
            print(f"❌ Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    main()
