import pandas as pd
import joblib
import os
import sys
from pathlib import Path
from app.ML.ml_models import BatsmanPerformanceModel, BowlerPerformanceModel, AllRounderPerformanceModel

def main():
    base_dir = '/workspaces/Cricket_website_ML_project'
    # Load your full dataset
    df = pd.read_csv(os.path.join(base_dir, 'data', 'flattened_cricket_data.csv'), low_memory=False)
    
    # Train Batsman Model (regression)
    batsman_model = BatsmanPerformanceModel(mode='regression')
    batsman_model.train_model(df)
    joblib.dump(batsman_model, os.path.join(base_dir, 'models', 'batsman_model.pkl'))
    
    # Train Bowler Model (classification for wickets >= 2)
    bowler_model = BowlerPerformanceModel(mode='classification')
    bowler_model.train_model(df)
    joblib.dump(bowler_model, os.path.join(base_dir, 'models', 'bowler_model.pkl'))
    
    # Train All-Rounder Model (classification)
    ar_model = AllRounderPerformanceModel()
    ar_model.train_model(df)
    joblib.dump(ar_model, os.path.join(base_dir, 'models', 'allrounder_model.pkl'))
    
    print("Specialized models trained and saved!")

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent.parent))
    main()
