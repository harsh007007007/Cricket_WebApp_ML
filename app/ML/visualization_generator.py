import plotly.express as px
import pandas as pd
import os
import json
from pathlib import Path

# Fix import with relative path
from app.ML.ml_models import CricketPerformanceModel

def generate_all_visualizations():
    # Handle the double workspaces path
    base_dir = '/workspaces/Cricket_website_ML_project'
    
    data_dir = os.path.join(base_dir, 'data')
    viz_dir = os.path.join(base_dir, 'app', 'static', 'viz')
    
    # Load data
    df = pd.read_csv(os.path.join(data_dir, 'flattened_cricket_data.csv'))
    with open(os.path.join(data_dir, 'structured_cricket_data.json')) as f:
        players = json.load(f)
    
    # Initialize and train model
    model = CricketPerformanceModel()
    model.train_models(df)
    
    # Create viz directory if not exists
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate cluster visualization
    features = model.prepare_features(df)
    fig = px.scatter(
        x=features['bat_Tests_avg'],
        y=features['bowl_Tests_avg'],
        color=model.kmeans.labels_,
        hover_name=df['NAME'],
        title="Player Clusters"
    )
    fig.write_html(os.path.join(viz_dir, 'clusters.html'))
    
    # Generate player-specific visualizations
    for player in players[:100]:  # First 100 players
        player_features = model.prepare_features(pd.DataFrame([player]))
        fig = px.bar(
            x=['Batting Avg', 'Bowling Avg'],
            y=[player_features['bat_Tests_avg'][0], player_features['bowl_Tests_avg'][0]],
            title=f"{player['NAME']} Profile"
        )
        # Clean player name for filename
        clean_name = "".join(c if c.isalnum() else "_" for c in player["NAME"])
        fig.write_html(os.path.join(viz_dir, f'player_{clean_name}_cluster.html'))

if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    generate_all_visualizations()