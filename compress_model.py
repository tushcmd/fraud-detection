# compress_model.py
import joblib
import xgboost as xgb

try:
    # Load your existing model
    print("Loading original model...")
    model = joblib.load('xgb_model.pkl')
    
    # Save it with maximum compression
    print("Compressing model...")
    joblib.dump(model, 'xgb_model_compressed.pkl', compress=9)
    
    print("Model compressed successfully!")
    
    # Verify the compression worked
    print("Verifying compressed model...")
    compressed_model = joblib.load('xgb_model_compressed.pkl')
    print("Compression verified!")
    
except Exception as e:
    print(f"An error occurred: {str(e)}")