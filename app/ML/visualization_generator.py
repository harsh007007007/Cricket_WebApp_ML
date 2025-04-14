#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import plotly.express as px
import pandas as pd
import json
from app.ML.ml_models import get_model_descriptions

def generate_visualizations():
    base_dir = '/workspaces/Cricket_website_ML_project'
    data_dir = os.path.join(base_dir, 'data')
    viz_dir = os.path.join(base_dir, 'app', 'static', 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(os.path.join(data_dir, 'flattened_cricket_data.csv'))
    with open(os.path.join(data_dir, 'structured_cricket_data.json')) as f:
        players = json.load(f)
    
    # Scatter plot for batsmen: Average vs. Strike Rate
    df['BATTING_Tests_Ave'] = pd.to_numeric(df['BATTING_Tests_Ave'], errors='coerce').fillna(0)
    df['BATTING_Tests_SR'] = pd.to_numeric(df['BATTING_Tests_SR'], errors='coerce').fillna(0)
    fig = px.scatter(
        df,
        x='BATTING_Tests_Ave',
        y='BATTING_Tests_SR',
        hover_name='NAME',
        title="Batsmen: Batting Average vs. Strike Rate"
    )
    fig.write_html(os.path.join(viz_dir, 'batsmen_scatter.html'))
    
    # Model explanation report
    model_desc = get_model_descriptions()
    html_content = "<html><head><title>Model Explanations</title></head><body>"
    html_content += "<h1>Model Explanations</h1>"
    for name, desc in model_desc.items():
        html_content += f"<h3>{name}</h3><p>{desc}</p>"
    html_content += "</body></html>"
    with open(os.path.join(viz_dir, 'model_explanations.html'), 'w') as f:
        f.write(html_content)

if __name__ == '__main__':
    generate_visualizations()
