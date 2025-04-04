from flask import render_template, send_from_directory
import pandas as pd
import json
import os
from pathlib import Path
from app import app  


# Load data
df = pd.read_csv(
    '/workspaces/Cricket_website_ML_project/data/flattened_cricket_data.csv',
    low_memory=False,
)
with open('/workspaces/Cricket_website_ML_project/data/structured_cricket_data.json') as f:
    players_json = json.load(f)

# Route to serve visualization files
@app.route('/viz/<path:filename>')
def viz_files(filename):
    return send_from_directory(os.path.join(app.static_folder, 'viz'), filename)

@app.route('/')
def home():
    # Default to test format when accessing root URL
    return format_analysis('test')

@app.route('/players')
def player_list():
    return render_template('players.html', players=players_json[:100])

@app.route('/player/<name>')
def player_profile(name):
    player = next((p for p in players_json if p['NAME'] == name), None)
    if not player:
        return render_template('404.html'), 404
    return render_template('player.html', player=player)

@app.route('/analysis/<format_type>')
def format_analysis(format_type):
    # Validate format type
    if format_type not in ['test', 'odi', 't20']:
        return render_template('404.html'), 404
    
    # Map format to visualization files with full paths
    viz_mapping = {
        'test': [
            'viz/top_test_batsmen.html',
            'viz/top_test_bowlers.html',
            'viz/test_allrounders.html'
        ],
        'odi': [
            'viz/top_odi_batsmen.html',
            'viz/odi_batting_impact.html',
            'viz/top_odi_bowlers.html',
            'viz/odi_allrounders.html',
            'viz/odi_role_analysis.html'
        ],
        't20': [
            'viz/top_t20_batsmen.html',
            'viz/t20_batting_impact.html',
            'viz/t20_strike_rates.html',
            'viz/top_t20_bowlers.html',
            'viz/t20_wicket_takers.html',
            'viz/t20_allrounders.html'
        ]
    }

    # Verify which files actually exist
    existing_files = []
    for viz_file in viz_mapping[format_type]:
        file_path = os.path.join(app.static_folder, viz_file)
        if os.path.exists(file_path):
            existing_files.append(viz_file.replace('viz/', ''))

    return render_template('analysis.html',
                        format=format_type,
                        viz_files=existing_files)

if __name__ == '__main__':
    # Ensure viz directory exists
    viz_dir = Path(__file__).parent / "app" / "static" / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    app.run(host='0.0.0.0', port=8000, debug=True)