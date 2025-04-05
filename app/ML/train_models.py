import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# Fix imports to use relative imports
from app.ML.ml_models import CricketPerformanceModel

def main():
    # Handle the double workspaces path
    base_dir = '/workspaces/Cricket_website_ML_project'
    
    # Load data
    df = pd.read_csv(os.path.join(base_dir, 'data', 'flattened_cricket_data.csv'), low_memory=False)
    
    # Initialize and train model
    model = CricketPerformanceModel()
    model.train_models(df)
    
    # Save model
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'performance_model.pkl'))
    print("Model trained and saved successfully")

if __name__ == '__main__':
    # Add parent directory to path when running directly
    sys.path.append(str(Path(__file__).parent.parent))
    main()