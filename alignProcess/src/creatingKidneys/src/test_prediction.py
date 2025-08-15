"""
Simple wrapper to run kidney prediction with better output handling
"""
import sys
import os
sys.path.append('.')

from kidneyMRIPrediction import KidneySlavesPipeline

def run_prediction(input_file):
    """Run kidney prediction on input file"""
    print("🧠 KIDNEY AI PREDICTION")
    print("="*50)
    print(f"📂 Input file: {input_file}")
    
    try:
        model_path = r"kidney_unet_model_best.pth"
        predictor = KidneySlavesPipeline(model_path)
        result = predictor.process_file(input_file)
        
        if result:
            print(f"✅ Prediction successful!")
            print(f"📂 Output: {result}")
        else:
            print("❌ Prediction failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = r"C:\Users\ftmen\Documents\mrialign\alignProcess\data\training\withoutROIwithMRI.mat"
    
    run_prediction(input_file)
